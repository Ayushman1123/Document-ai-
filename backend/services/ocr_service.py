"""
OCR Service using PaddleOCR for multilingual text extraction
"""
import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class OCRService:
    """
    Multi-language OCR service using PaddleOCR
    Supports: English, Hindi, Gujarati, Tamil, Telugu, Marathi
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.ocr_engine = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize PaddleOCR engine with multilingual support"""
        try:
            from paddleocr import PaddleOCR
            
            # Initialize with multilingual support
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # Primary language
                use_gpu=self.use_gpu,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                drop_score=0.3,
            )
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.warning("PaddleOCR not installed. Using fallback OCR.")
            self.ocr_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr_engine = None
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from image with bounding boxes
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            Dict containing:
                - raw_text: Full extracted text
                - text_blocks: List of text blocks with bounding boxes
                - confidence: Overall confidence score
        """
        if self.ocr_engine is None:
            return self._fallback_ocr(image)
        
        try:
            # Run OCR
            result = self.ocr_engine.ocr(image, cls=True)
            
            if not result or not result[0]:
                return {
                    "raw_text": "",
                    "text_blocks": [],
                    "confidence": 0.0
                }
            
            text_blocks = []
            full_text_parts = []
            confidence_scores = []
            
            for line in result[0]:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert bbox to standard format [x1, y1, x2, y2]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                text_block = {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": int(min(x_coords)),
                        "y1": int(min(y_coords)),
                        "x2": int(max(x_coords)),
                        "y2": int(max(y_coords))
                    },
                    "polygon": [[int(p[0]), int(p[1])] for p in bbox]
                }
                
                text_blocks.append(text_block)
                full_text_parts.append(text)
                confidence_scores.append(confidence)
            
            # Sort text blocks by position (top to bottom, left to right)
            text_blocks = sorted(text_blocks, key=lambda x: (x["bbox"]["y1"], x["bbox"]["x1"]))
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                "raw_text": "\n".join(full_text_parts),
                "text_blocks": text_blocks,
                "confidence": float(avg_confidence)
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "raw_text": "",
                "text_blocks": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _fallback_ocr(self, image: np.ndarray) -> Dict:
        """Fallback OCR using basic method when PaddleOCR is unavailable"""
        logger.warning("Using fallback OCR - results may be less accurate")
        return {
            "raw_text": "[OCR not available - please install PaddleOCR]",
            "text_blocks": [],
            "confidence": 0.0
        }
    
    def extract_structured_text(self, image: np.ndarray) -> Dict:
        """
        Extract text with layout-aware structure
        Groups text into logical sections (header, body, footer)
        """
        ocr_result = self.extract_text(image)
        
        if not ocr_result["text_blocks"]:
            return ocr_result
        
        height = image.shape[0]
        
        # Classify blocks by vertical position
        header_blocks = []
        body_blocks = []
        footer_blocks = []
        
        for block in ocr_result["text_blocks"]:
            y_center = (block["bbox"]["y1"] + block["bbox"]["y2"]) / 2
            relative_pos = y_center / height
            
            if relative_pos < 0.2:
                header_blocks.append(block)
            elif relative_pos > 0.8:
                footer_blocks.append(block)
            else:
                body_blocks.append(block)
        
        ocr_result["structured"] = {
            "header": header_blocks,
            "body": body_blocks,
            "footer": footer_blocks
        }
        
        return ocr_result
    
    def find_text_pattern(self, image: np.ndarray, patterns: List[str]) -> List[Dict]:
        """
        Find specific text patterns in the image
        
        Args:
            image: Input image
            patterns: List of regex patterns to search for
            
        Returns:
            List of matches with text and bounding boxes
        """
        import re
        
        ocr_result = self.extract_text(image)
        matches = []
        
        for block in ocr_result["text_blocks"]:
            text = block["text"]
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append({
                        "pattern": pattern,
                        "matched_text": text,
                        "bbox": block["bbox"],
                        "confidence": block["confidence"]
                    })
        
        return matches
