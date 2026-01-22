"""
Main Document Processing Pipeline
Orchestrates OCR, Vision, and Field Extraction services
"""
import os
import uuid
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import json
import time

from .ocr_service import OCRService
from .vision_service import VisionService
from .field_extractor import FieldExtractor

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    End-to-end document processing pipeline for invoice field extraction
    
    Pipeline stages:
    1. Document Ingestion (PDF/Image loading and preprocessing)
    2. OCR Extraction (Text and layout)
    3. Vision Analysis (Signature/Stamp detection)
    4. Field Extraction (Structured data extraction)
    5. Post-processing and Validation
    6. Output Generation
    """
    
    def __init__(self, 
                 use_gpu: bool = False,
                 use_llm: bool = True,
                 openai_api_key: str = None,
                 google_api_key: str = None):
        """
        Initialize document processor
        
        Args:
            use_gpu: Whether to use GPU for processing
            use_llm: Whether to use LLM for field extraction
            openai_api_key: OpenAI API key
            google_api_key: Google API key
        """
        self.use_gpu = use_gpu
        self.use_llm = use_llm
        
        # Initialize services
        self.ocr_service = OCRService(use_gpu=use_gpu)
        self.vision_service = VisionService()
        self.field_extractor = FieldExtractor(
            use_llm=use_llm,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key
        )
        
        logger.info("DocumentProcessor initialized")
    
    def process_document(self, 
                        file_path: str,
                        output_dir: str = None) -> Dict:
        """
        Process a single document and extract all fields
        
        Args:
            file_path: Path to PDF or image file
            output_dir: Directory to save outputs
            
        Returns:
            Dict containing extracted fields and metadata
        """
        start_time = time.time()
        
        # Generate document ID
        doc_id = str(uuid.uuid4())[:8]
        
        result = {
            "document_id": doc_id,
            "file_name": os.path.basename(file_path),
            "timestamp": datetime.now().isoformat(),
            "status": "processing",
            "fields": {},
            "detections": {},
            "metadata": {},
            "errors": []
        }
        
        try:
            # Step 1: Load and preprocess document
            logger.info(f"Processing document: {file_path}")
            images = self._load_document(file_path)
            
            if not images:
                result["status"] = "error"
                result["errors"].append("Failed to load document")
                return result
            
            result["metadata"]["page_count"] = len(images)
            result["metadata"]["image_dimensions"] = [
                {"width": img.shape[1], "height": img.shape[0]} 
                for img in images
            ]
            
            # Process each page (for multi-page PDFs)
            all_ocr_text = []
            all_text_blocks = []
            all_detections = []
            
            for page_idx, image in enumerate(images):
                logger.info(f"Processing page {page_idx + 1}/{len(images)}")
                
                # Step 2: OCR Extraction
                ocr_result = self.ocr_service.extract_structured_text(image)
                all_ocr_text.append(ocr_result["raw_text"])
                all_text_blocks.extend(ocr_result["text_blocks"])
                
                # Step 3: Vision Analysis (Signature/Stamp Detection)
                vision_result = self.vision_service.detect_signatures_stamps(image)
                
                # Adjust detection coordinates for multi-page
                if page_idx > 0:
                    offset = sum(img.shape[0] for img in images[:page_idx])
                    for category in ["signature", "stamp"]:
                        for det in vision_result[category]["detections"]:
                            det["bbox"]["y1"] += offset
                            det["bbox"]["y2"] += offset
                            det["page"] = page_idx
                
                all_detections.append({
                    "page": page_idx,
                    "signature": vision_result["signature"],
                    "stamp": vision_result["stamp"]
                })
            
            # Combine OCR from all pages
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_ocr_text)
            
            # Step 4: Field Extraction
            extracted_fields = self.field_extractor.extract_fields(
                ocr_text=combined_text,
                text_blocks=all_text_blocks,
                image=images[0]  # Use first page for LLM vision
            )
            
            # Step 5: Compile Signature/Stamp Results
            signature_present = any(d["signature"]["present"] for d in all_detections)
            stamp_present = any(d["stamp"]["present"] for d in all_detections)
            
            signature_detections = []
            stamp_detections = []
            
            for d in all_detections:
                signature_detections.extend(d["signature"]["detections"])
                stamp_detections.extend(d["stamp"]["detections"])
            
            # Step 6: Generate Output
            result["fields"] = {
                "dealer_name": {
                    "value": extracted_fields["dealer_name"]["value"],
                    "confidence": extracted_fields["dealer_name"]["confidence"],
                    "extraction_method": extracted_fields["dealer_name"]["method"]
                },
                "model_name": {
                    "value": extracted_fields["model_name"]["value"],
                    "confidence": extracted_fields["model_name"]["confidence"],
                    "extraction_method": extracted_fields["model_name"]["method"]
                },
                "horse_power": {
                    "value": extracted_fields["horse_power"]["value"],
                    "confidence": extracted_fields["horse_power"]["confidence"],
                    "extraction_method": extracted_fields["horse_power"]["method"]
                },
                "asset_cost": {
                    "value": extracted_fields["asset_cost"]["value"],
                    "confidence": extracted_fields["asset_cost"]["confidence"],
                    "extraction_method": extracted_fields["asset_cost"]["method"]
                },
                "dealer_signature": {
                    "present": signature_present,
                    "confidence": max((d["signature"]["confidence"] for d in all_detections), default=0.0),
                    "bounding_boxes": [d["bbox"] for d in signature_detections]
                },
                "dealer_stamp": {
                    "present": stamp_present,
                    "confidence": max((d["stamp"]["confidence"] for d in all_detections), default=0.0),
                    "bounding_boxes": [d["bbox"] for d in stamp_detections]
                }
            }
            
            # Calculate overall confidence
            confidences = [
                result["fields"]["dealer_name"]["confidence"],
                result["fields"]["model_name"]["confidence"],
                result["fields"]["horse_power"]["confidence"],
                result["fields"]["asset_cost"]["confidence"],
                result["fields"]["dealer_signature"]["confidence"],
                result["fields"]["dealer_stamp"]["confidence"]
            ]
            result["metadata"]["overall_confidence"] = np.mean([c for c in confidences if c > 0])
            
            # Validate against master data
            validation = self.field_extractor.validate_against_master(extracted_fields)
            result["validation"] = validation
            
            # Calculate processing time and cost
            processing_time = time.time() - start_time
            result["metadata"]["processing_time_seconds"] = round(processing_time, 2)
            result["metadata"]["estimated_cost_usd"] = self._estimate_cost(
                len(images), 
                len(combined_text),
                use_llm=self.use_llm
            )
            
            result["status"] = "completed"
            
            # Save outputs
            if output_dir:
                self._save_outputs(result, images, output_dir, doc_id)
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            result["status"] = "error"
            result["errors"].append(str(e))
        
        return result
    
    def _load_document(self, file_path: str) -> List[np.ndarray]:
        """Load document (PDF or image) and return as list of images"""
        images = []
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                # Convert PDF to images
                try:
                    from pdf2image import convert_from_path
                    pil_images = convert_from_path(file_path, dpi=200)
                    for pil_img in pil_images:
                        images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
                except ImportError:
                    logger.error("pdf2image not installed. Cannot process PDF.")
                    return []
            else:
                # Load image directly
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
            
            # Preprocess images
            images = [self._preprocess_image(img) for img in images]
            
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
        
        return images
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Resize if too large
        max_dim = 2048
        height, width = image.shape[:2]
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def _estimate_cost(self, page_count: int, text_length: int, use_llm: bool) -> float:
        """Estimate processing cost per document"""
        # Base cost for OCR processing (local, effectively free)
        ocr_cost = 0.0
        
        # Vision processing cost (local, effectively free)
        vision_cost = 0.0
        
        # LLM cost if used
        llm_cost = 0.0
        if use_llm:
            # Estimate based on token usage
            # GPT-4 Vision: ~$0.01 per 1000 tokens input, ~$0.03 per 1000 tokens output
            tokens_in = text_length // 4 + 1000  # Text + image tokens
            tokens_out = 200  # Average response
            llm_cost = (tokens_in * 0.01 + tokens_out * 0.03) / 1000
        
        total_cost = ocr_cost + vision_cost + llm_cost
        
        return round(total_cost, 4)
    
    def _save_outputs(self, result: Dict, images: List[np.ndarray], 
                     output_dir: str, doc_id: str):
        """Save processing outputs to files"""
        output_path = Path(output_dir) / doc_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON result
        with open(output_path / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save annotated images
        for idx, image in enumerate(images):
            # Draw bounding boxes
            annotated = image.copy()
            
            # Draw signature boxes (green)
            for bbox in result["fields"]["dealer_signature"]["bounding_boxes"]:
                cv2.rectangle(
                    annotated,
                    (bbox["x1"], bbox["y1"]),
                    (bbox["x2"], bbox["y2"]),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    annotated, "Signature",
                    (bbox["x1"], bbox["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            # Draw stamp boxes (blue)
            for bbox in result["fields"]["dealer_stamp"]["bounding_boxes"]:
                cv2.rectangle(
                    annotated,
                    (bbox["x1"], bbox["y1"]),
                    (bbox["x2"], bbox["y2"]),
                    (255, 0, 0), 2
                )
                cv2.putText(
                    annotated, "Stamp",
                    (bbox["x1"], bbox["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
            
            cv2.imwrite(str(output_path / f"annotated_page_{idx + 1}.jpg"), annotated)
        
        logger.info(f"Outputs saved to {output_path}")
    
    def process_batch(self, file_paths: List[str], output_dir: str = None) -> List[Dict]:
        """Process multiple documents in batch"""
        results = []
        
        for idx, file_path in enumerate(file_paths):
            logger.info(f"Processing document {idx + 1}/{len(file_paths)}")
            result = self.process_document(file_path, output_dir)
            results.append(result)
        
        # Calculate batch statistics
        successful = sum(1 for r in results if r["status"] == "completed")
        failed = len(results) - successful
        
        batch_summary = {
            "total_documents": len(results),
            "successful": successful,
            "failed": failed,
            "average_processing_time": np.mean([
                r["metadata"].get("processing_time_seconds", 0) 
                for r in results if r["status"] == "completed"
            ]),
            "total_estimated_cost": sum([
                r["metadata"].get("estimated_cost_usd", 0) 
                for r in results if r["status"] == "completed"
            ])
        }
        
        logger.info(f"Batch processing complete: {batch_summary}")
        
        return results


class DocumentMetrics:
    """Calculate evaluation metrics for document extraction"""
    
    @staticmethod
    def calculate_document_level_accuracy(predictions: List[Dict], 
                                         ground_truth: List[Dict]) -> Dict:
        """
        Calculate Document-Level Accuracy (DLA)
        A document is accurate if ALL fields are correctly extracted
        """
        correct_docs = 0
        field_accuracies = {
            "dealer_name": 0,
            "model_name": 0,
            "horse_power": 0,
            "asset_cost": 0,
            "dealer_signature": 0,
            "dealer_stamp": 0
        }
        
        for pred, gt in zip(predictions, ground_truth):
            all_correct = True
            
            # Check dealer name (fuzzy match >= 90%)
            if DocumentMetrics._fuzzy_match(
                pred["fields"]["dealer_name"]["value"],
                gt.get("dealer_name")
            ) >= 90:
                field_accuracies["dealer_name"] += 1
            else:
                all_correct = False
            
            # Check model name (exact match)
            if DocumentMetrics._exact_match(
                pred["fields"]["model_name"]["value"],
                gt.get("model_name")
            ):
                field_accuracies["model_name"] += 1
            else:
                all_correct = False
            
            # Check horse power (±5% tolerance)
            if DocumentMetrics._numeric_match(
                pred["fields"]["horse_power"]["value"],
                gt.get("horse_power"),
                tolerance=0.05
            ):
                field_accuracies["horse_power"] += 1
            else:
                all_correct = False
            
            # Check asset cost (±5% tolerance)
            if DocumentMetrics._numeric_match(
                pred["fields"]["asset_cost"]["value"],
                gt.get("asset_cost"),
                tolerance=0.05
            ):
                field_accuracies["asset_cost"] += 1
            else:
                all_correct = False
            
            # Check signature (presence + IoU >= 0.5)
            if DocumentMetrics._detection_match(
                pred["fields"]["dealer_signature"],
                gt.get("dealer_signature", {})
            ):
                field_accuracies["dealer_signature"] += 1
            else:
                all_correct = False
            
            # Check stamp (presence + IoU >= 0.5)
            if DocumentMetrics._detection_match(
                pred["fields"]["dealer_stamp"],
                gt.get("dealer_stamp", {})
            ):
                field_accuracies["dealer_stamp"] += 1
            else:
                all_correct = False
            
            if all_correct:
                correct_docs += 1
        
        total = len(predictions)
        
        return {
            "document_level_accuracy": correct_docs / total if total > 0 else 0,
            "field_accuracies": {
                k: v / total if total > 0 else 0 
                for k, v in field_accuracies.items()
            },
            "total_documents": total,
            "correct_documents": correct_docs
        }
    
    @staticmethod
    def _fuzzy_match(pred: str, gt: str) -> float:
        """Calculate fuzzy match score"""
        from fuzzywuzzy import fuzz
        
        if pred is None or gt is None:
            return 0 if pred != gt else 100
        
        return fuzz.ratio(str(pred).upper(), str(gt).upper())
    
    @staticmethod
    def _exact_match(pred: str, gt: str) -> bool:
        """Check exact match (case-insensitive)"""
        if pred is None and gt is None:
            return True
        if pred is None or gt is None:
            return False
        
        return str(pred).upper().strip() == str(gt).upper().strip()
    
    @staticmethod
    def _numeric_match(pred: float, gt: float, tolerance: float = 0.05) -> bool:
        """Check numeric match within tolerance"""
        if pred is None and gt is None:
            return True
        if pred is None or gt is None:
            return False
        
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            
            if gt_val == 0:
                return pred_val == 0
            
            return abs(pred_val - gt_val) / gt_val <= tolerance
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def _detection_match(pred: Dict, gt: Dict, iou_threshold: float = 0.5) -> bool:
        """Check detection match (presence + IoU)"""
        pred_present = pred.get("present", False)
        gt_present = gt.get("present", False)
        
        if pred_present != gt_present:
            return False
        
        if not pred_present:  # Both are False
            return True
        
        # Check IoU
        pred_boxes = pred.get("bounding_boxes", [])
        gt_boxes = gt.get("bounding_boxes", [])
        
        if not gt_boxes:
            return bool(pred_boxes)  # GT has no boxes, accept any detection
        
        for gt_box in gt_boxes:
            for pred_box in pred_boxes:
                iou = DocumentMetrics._calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    return True
        
        return False
    
    @staticmethod
    def _calculate_iou(box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union"""
        x1_inter = max(box1["x1"], box2["x1"])
        y1_inter = max(box1["y1"], box2["y1"])
        x2_inter = min(box1["x2"], box2["x2"])
        y2_inter = min(box1["y2"], box2["y2"])
        
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
