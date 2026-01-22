"""
Configuration settings for Document AI System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (set these in .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model settings
class ModelConfig:
    # OCR Settings
    OCR_LANG = ["en", "hi", "mr", "ta", "te", "gu"]  # Multilingual support
    OCR_USE_GPU = False
    OCR_DET_MODEL_DIR = str(MODELS_DIR / "det")
    OCR_REC_MODEL_DIR = str(MODELS_DIR / "rec")
    OCR_CLS_MODEL_DIR = str(MODELS_DIR / "cls")
    
    # YOLO Settings for Signature/Stamp Detection
    YOLO_MODEL_PATH = str(MODELS_DIR / "signature_stamp_detector.pt")
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    
    # LLM Settings
    LLM_MODEL = "gpt-4-vision-preview"  # or "gemini-pro-vision"
    LLM_MAX_TOKENS = 2000
    LLM_TEMPERATURE = 0.1
    
    # Processing Settings
    MAX_IMAGE_SIZE = (2048, 2048)
    PDF_DPI = 200
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.9
    MEDIUM_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.5

class EvaluationConfig:
    # Fuzzy matching threshold for dealer name
    FUZZY_MATCH_THRESHOLD = 90
    
    # Numeric tolerance for HP/Cost
    NUMERIC_TOLERANCE = 0.05  # 5%
    
    # IoU threshold for signature/stamp detection
    IOU_THRESHOLD = 0.5

# API Settings
class APIConfig:
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5500"]
