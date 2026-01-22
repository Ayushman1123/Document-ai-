"""
Document AI System - FastAPI Backend
Main application entry point
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import uuid
import json
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr
import uvicorn

from config import APIConfig, ModelConfig, BASE_DIR, UPLOAD_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document AI System",
    description="Intelligent Document AI for Field Extraction from Invoices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ExtractionResult(BaseModel):
    document_id: str
    file_name: str
    status: str
    fields: dict
    metadata: dict
    validation: Optional[dict] = None
    explanation: Optional[dict] = None
    agentic_reasoning: Optional[dict] = None

class FeedbackRequest(BaseModel):
    document_id: str
    field_name: str
    predicted_value: Optional[str] = None
    correct_value: str
    extraction_method: Optional[str] = None

class BatchProcessRequest(BaseModel):
    document_ids: List[str]

class OrchestrationResult(BaseModel):
    document_id: str
    file_name: str
    status: str
    fields: dict
    overall_confidence: float
    agentic_reasoning: dict
    agents_used: List[str]
    reasoning_steps: int

# Authentication Models
class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: bool = False

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    password: str
    enable_mfa: bool = True

class MFAVerifyRequest(BaseModel):
    mfa_token: str
    otp_code: str

class ResendOTPRequest(BaseModel):
    mfa_token: str

# Store for tracking processing jobs
processing_jobs = {}

# Store for extraction history
extraction_history = []

# Lazy initialization of services
_document_processor = None
_agentic_ai = None
_agentic_coordinator = None
_auth_service = None

def get_auth_service():
    """Lazy initialization of auth service"""
    global _auth_service
    if _auth_service is None:
        try:
            from services.auth_service import AuthService
            _auth_service = AuthService()
            logger.info("AuthService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AuthService: {e}")
            _auth_service = None
    return _auth_service


def get_document_processor():
    """Lazy initialization of document processor"""
    global _document_processor
    if _document_processor is None:
        try:
            from services.document_processor import DocumentProcessor
            openai_key = os.getenv("OPENAI_API_KEY", "")
            google_key = os.getenv("GOOGLE_API_KEY", "")
            _document_processor = DocumentProcessor(
                use_gpu=False,
                use_llm=bool(openai_key or google_key),
                openai_api_key=openai_key,
                google_api_key=google_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {e}")
            raise HTTPException(status_code=500, detail="Service initialization failed")
    return _document_processor

def get_agentic_ai():
    """Lazy initialization of agentic AI"""
    global _agentic_ai
    if _agentic_ai is None:
        try:
            from services.agentic_ai import AgenticAI
            openai_key = os.getenv("OPENAI_API_KEY", "")
            google_key = os.getenv("GOOGLE_API_KEY", "")
            _agentic_ai = AgenticAI(
                openai_api_key=openai_key,
                google_api_key=google_key,
                feedback_store_path=str(BASE_DIR / "data" / "feedback")
            )
        except Exception as e:
            logger.error(f"Failed to initialize AgenticAI: {e}")
            _agentic_ai = None
    return _agentic_ai

def get_agentic_coordinator():
    """Lazy initialization of agentic coordinator"""
    global _agentic_coordinator
    if _agentic_coordinator is None:
        try:
            from services.coordinator import AgenticCoordinator
            from services.ocr_service import OCRService
            from services.vision_service import VisionService
            from services.field_extractor import FieldExtractor
            
            ocr_service = OCRService(use_gpu=False)
            vision_service = VisionService()
            field_extractor = FieldExtractor(use_llm=False)
            
            _agentic_coordinator = AgenticCoordinator(
                ocr_service=ocr_service,
                vision_service=vision_service,
                field_extractor=field_extractor
            )
            logger.info("AgenticCoordinator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticCoordinator: {e}")
            _agentic_coordinator = None
    return _agentic_coordinator

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Document AI System",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "ocr": "available",
            "vision": "available",
            "field_extraction": "available",
            "agentic_ai": "available" if get_agentic_ai() else "unavailable",
            "agentic_orchestrator": "available" if get_agentic_coordinator() else "unavailable",
            "authentication": "available" if get_auth_service() else "unavailable"
        },
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR)
    }

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """
    Register a new user
    
    If MFA is enabled, returns an MFA token for verification.
    Otherwise, creates the account immediately.
    """
    auth = get_auth_service()
    if not auth:
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    result = auth.register(
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        last_name=request.last_name,
        phone=request.phone,
        enable_mfa=request.enable_mfa
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))
    
    return result

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """
    Authenticate user
    
    If MFA is enabled for the user, returns an MFA token.
    Otherwise, returns an access token directly.
    """
    auth = get_auth_service()
    if not auth:
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    result = auth.login(
        email=request.email,
        password=request.password,
        remember_me=request.remember_me
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error", "Login failed"))
    
    return result

@app.post("/api/auth/verify-mfa")
async def verify_mfa(request: MFAVerifyRequest):
    """
    Verify MFA code and complete login/registration
    
    Returns access token on successful verification.
    """
    auth = get_auth_service()
    if not auth:
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    result = auth.verify_mfa(
        mfa_token=request.mfa_token,
        otp_code=request.otp_code
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error", "Verification failed"))
    
    return result

@app.post("/api/auth/resend-otp")
async def resend_otp(request: ResendOTPRequest):
    """Resend OTP code for MFA verification"""
    auth = get_auth_service()
    if not auth:
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    result = auth.resend_otp(mfa_token=request.mfa_token)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to resend"))
    
    return result

@app.get("/api/auth/validate")
async def validate_token(authorization: str = Header(None)):
    """Validate access token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    # Extract token from "Bearer <token>" format
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    
    auth = get_auth_service()
    if not auth:
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    result = auth.validate_token(token)
    
    if not result.get("valid"):
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid token"))
    
    return result

@app.post("/api/auth/logout")
async def logout(authorization: str = Header(None)):
    """Logout and invalidate session"""
    if not authorization:
        return {"success": True, "message": "Already logged out"}
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    
    auth = get_auth_service()
    if auth:
        auth.logout(token)
    
    return {"success": True, "message": "Logged out successfully"}

# ==================== DOCUMENT PROCESSING ENDPOINTS ====================

@app.post("/api/extract/orchestrated", response_model=OrchestrationResult)
async def extract_with_orchestration(file: UploadFile = File(...)):
    """
    Extract fields using multi-agent orchestration
    
    This endpoint uses specialized agents:
    - OCR Agent: Extracts text using PaddleOCR
    - Vision Agent: Detects signatures and stamps
    - Extractor Agent: Extracts structured fields
    - Validator Agent: Validates and corrects results
    
    Returns full reasoning chain for explainability
    """
    import cv2
    import numpy as np
    
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type")
    
    doc_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{doc_id}{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved for orchestration: {upload_path}")
        
        # Load image
        if file_ext == '.pdf':
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(upload_path), dpi=200)
                image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            except ImportError:
                raise HTTPException(status_code=500, detail="pdf2image not installed")
        else:
            image = cv2.imread(str(upload_path))
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load image")
        
        # Get orchestrator
        coordinator = get_agentic_coordinator()
        if coordinator is None:
            raise HTTPException(status_code=500, detail="Orchestrator not available")
        
        # Process with orchestration
        result = await coordinator.process_document(doc_id, image)
        
        return OrchestrationResult(
            document_id=doc_id,
            file_name=file.filename,
            status=result.get("status", "completed"),
            fields=result.get("fields", {}),
            overall_confidence=result.get("overall_confidence", 0),
            agentic_reasoning=result.get("agentic_reasoning", {}),
            agents_used=result.get("agentic_reasoning", {}).get("agents_used", []),
            reasoning_steps=result.get("agentic_reasoning", {}).get("total_reasoning_steps", 0)
        )
        
    except Exception as e:
        logger.error(f"Orchestrated extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orchestration/{document_id}/reasoning")
async def get_reasoning(document_id: str):
    """Get detailed reasoning explanation for a processed document"""
    coordinator = get_agentic_coordinator()
    if coordinator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not available")
    
    explanation = coordinator.get_reasoning_explanation(document_id)
    state = coordinator.get_workflow_state(document_id)
    
    if state is None:
        raise HTTPException(status_code=404, detail="No workflow found for this document")
    
    return {
        "document_id": document_id,
        "current_phase": state.current_phase,
        "reasoning_log": state.reasoning_log,
        "explanation_markdown": explanation
    }

@app.post("/api/extract", response_model=ExtractionResult)
async def extract_fields(
    file: UploadFile = File(...),
    use_agentic: bool = Query(False, description="Use agentic AI for intelligent extraction")
):
    """
    Extract fields from uploaded invoice document
    
    Supports:
    - PDF files
    - Image files (PNG, JPG, JPEG, TIFF)
    
    Returns extracted fields with confidence scores
    """
    # Validate file type
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate document ID
    doc_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{doc_id}{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved: {upload_path}")
        
        # Process document
        processor = get_document_processor()
        result = processor.process_document(
            str(upload_path),
            str(OUTPUT_DIR)
        )
        
        # Add agentic AI enhancements if requested
        if use_agentic:
            agentic = get_agentic_ai()
            if agentic:
                explanation = agentic.provide_explanation(result)
                result["explanation"] = explanation
        
        return ExtractionResult(
            document_id=result["document_id"],
            file_name=result["file_name"],
            status=result["status"],
            fields=result["fields"],
            metadata=result["metadata"],
            validation=result.get("validation"),
            explanation=result.get("explanation")
        )
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded file after processing
        if upload_path.exists():
            # Keep for debugging, remove in production
            pass

@app.post("/api/extract/batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process multiple documents in batch
    Returns job ID for tracking progress
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Save all files
    file_paths = []
    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_id = str(uuid.uuid4())[:8]
        upload_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_paths.append(str(upload_path))
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": "queued",
        "total": len(files),
        "processed": 0,
        "results": [],
        "started_at": datetime.now().isoformat()
    }
    
    # Process in background
    background_tasks.add_task(process_batch_job, job_id, file_paths)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total_files": len(files),
        "message": "Batch processing started. Use /api/batch/{job_id} to check status."
    }

async def process_batch_job(job_id: str, file_paths: List[str]):
    """Background task for batch processing"""
    processing_jobs[job_id]["status"] = "processing"
    
    processor = get_document_processor()
    results = []
    
    for idx, file_path in enumerate(file_paths):
        try:
            result = processor.process_document(file_path, str(OUTPUT_DIR))
            results.append(result)
        except Exception as e:
            results.append({
                "file_name": os.path.basename(file_path),
                "status": "error",
                "error": str(e)
            })
        
        processing_jobs[job_id]["processed"] = idx + 1
    
    processing_jobs[job_id]["status"] = "completed"
    processing_jobs[job_id]["results"] = results
    processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()

@app.get("/api/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get status of batch processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit correction feedback for learning
    Used to improve extraction accuracy over time
    """
    agentic = get_agentic_ai()
    
    if agentic:
        agentic.process_feedback(
            document_id=feedback.document_id,
            field_name=feedback.field_name,
            predicted_value=feedback.predicted_value,
            correct_value=feedback.correct_value,
            extraction_method=feedback.extraction_method or "unknown"
        )
        
        return {"status": "success", "message": "Feedback recorded"}
    else:
        return {"status": "warning", "message": "Agentic AI not available, feedback not recorded"}

@app.get("/api/results/{document_id}")
async def get_result(document_id: str):
    """Get extraction result by document ID"""
    result_path = OUTPUT_DIR / document_id / "result.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/results/{document_id}/annotated")
async def get_annotated_image(document_id: str, page: int = 1):
    """Get annotated image with bounding boxes"""
    image_path = OUTPUT_DIR / document_id / f"annotated_page_{page}.jpg"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Annotated image not found")
    
    return FileResponse(str(image_path), media_type="image/jpeg")

@app.get("/api/stats")
async def get_stats():
    """Get processing statistics"""
    # Count processed documents
    output_dirs = list(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
    
    total_processed = len([d for d in output_dirs if d.is_dir()])
    
    # Calculate average metrics
    processing_times = []
    confidences = []
    field_confidences = {
        "dealer_name": [],
        "model_name": [],
        "horse_power": [],
        "asset_cost": [],
        "dealer_signature": [],
        "dealer_stamp": []
    }
    
    for dir_path in output_dirs:
        if dir_path.is_dir():
            result_file = dir_path / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        result = json.load(f)
                        if "metadata" in result:
                            if "processing_time_seconds" in result["metadata"]:
                                processing_times.append(result["metadata"]["processing_time_seconds"])
                            if "overall_confidence" in result["metadata"]:
                                confidences.append(result["metadata"]["overall_confidence"])
                        
                        # Extract field-level confidence
                        if "fields" in result:
                            for field_name, field_data in result["fields"].items():
                                if field_name in field_confidences and isinstance(field_data, dict):
                                    conf = field_data.get("confidence", 0)
                                    field_confidences[field_name].append(conf)
                except:
                    pass
    
    # Calculate field averages
    field_accuracy = {}
    for field_name, confs in field_confidences.items():
        field_accuracy[field_name] = sum(confs) / len(confs) if confs else 0
    
    return {
        "total_documents_processed": total_processed,
        "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"]),
        "field_accuracy": field_accuracy
    }

@app.get("/api/history")
async def get_extraction_history(limit: int = Query(50, description="Maximum number of items")):
    """Get extraction history"""
    output_dirs = list(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else []
    
    history = []
    for dir_path in sorted(output_dirs, reverse=True)[:limit]:
        if dir_path.is_dir():
            result_file = dir_path / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        result = json.load(f)
                        history.append({
                            "document_id": result.get("document_id", dir_path.name),
                            "file_name": result.get("file_name", "Unknown"),
                            "status": result.get("status", "completed"),
                            "overall_confidence": result.get("metadata", {}).get("overall_confidence", 0),
                            "processing_time": result.get("metadata", {}).get("processing_time_seconds", 0),
                            "timestamp": result.get("metadata", {}).get("processed_at", "")
                        })
                except:
                    pass
    
    return {"history": history, "total": len(history)}

@app.delete("/api/results/{document_id}")
async def delete_result(document_id: str):
    """Delete processing result"""
    result_path = OUTPUT_DIR / document_id
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    shutil.rmtree(result_path)
    
    return {"status": "success", "message": f"Result {document_id} deleted"}

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=APIConfig.DEBUG
    )

