"""
Vision Service for Signature and Stamp Detection using YOLO
"""
import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class VisionService:
    """
    Computer Vision service for detecting signatures and stamps in documents
    Uses YOLOv8 for object detection with custom-trained weights
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.classes = ["signature", "stamp"]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model for signature/stamp detection"""
        try:
            from ultralytics import YOLO
            
            if self.model_path and os.path.exists(self.model_path):
                # Load custom trained model
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom YOLO model from {self.model_path}")
            else:
                # Use pretrained YOLOv8 as base (will be fine-tuned)
                self.model = YOLO('yolov8n.pt')
                logger.info("Loaded pretrained YOLOv8n model (needs fine-tuning for signatures/stamps)")
                
        except ImportError:
            logger.warning("Ultralytics not installed. Using heuristic detection.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            self.model = None
    
    def detect_signatures_stamps(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect signatures and stamps in document image
        
        Args:
            image: numpy array (BGR format)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dict containing signature and stamp detections with bounding boxes
        """
        result = {
            "signature": {
                "present": False,
                "detections": [],
                "confidence": 0.0
            },
            "stamp": {
                "present": False,
                "detections": [],
                "confidence": 0.0
            }
        }
        
        if self.model is not None:
            result = self._detect_with_yolo(image, confidence_threshold)
        
        # Use heuristic detection as fallback or enhancement
        heuristic_result = self._detect_with_heuristics(image)
        
        # Merge results
        result = self._merge_detections(result, heuristic_result)
        
        return result
    
    def _detect_with_yolo(self, image: np.ndarray, confidence_threshold: float) -> Dict:
        """Use YOLO model for detection"""
        result = {
            "signature": {"present": False, "detections": [], "confidence": 0.0},
            "stamp": {"present": False, "detections": [], "confidence": 0.0}
        }
        
        try:
            predictions = self.model.predict(image, conf=confidence_threshold, verbose=False)
            
            for pred in predictions:
                if pred.boxes is not None:
                    for box in pred.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            "bbox": {
                                "x1": int(bbox[0]),
                                "y1": int(bbox[1]),
                                "x2": int(bbox[2]),
                                "y2": int(bbox[3])
                            },
                            "confidence": conf
                        }
                        
                        class_name = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"
                        
                        if class_name in result:
                            result[class_name]["detections"].append(detection)
                            result[class_name]["present"] = True
                            result[class_name]["confidence"] = max(
                                result[class_name]["confidence"], conf
                            )
                            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
        
        return result
    
    def _detect_with_heuristics(self, image: np.ndarray) -> Dict:
        """
        Heuristic-based detection for signatures and stamps
        Uses image processing techniques to find potential regions
        """
        result = {
            "signature": {"present": False, "detections": [], "confidence": 0.0},
            "stamp": {"present": False, "detections": [], "confidence": 0.0}
        }
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            
            # Focus on bottom portion of document (where signatures usually are)
            bottom_region = gray[int(height * 0.6):, :]
            
            # Detect signature regions (look for handwritten strokes)
            signature_regions = self._detect_signature_regions(bottom_region, int(height * 0.6))
            
            # Detect stamp regions (look for circular or rectangular colored regions)
            stamp_regions = self._detect_stamp_regions(image)
            
            if signature_regions:
                result["signature"]["present"] = True
                result["signature"]["detections"] = signature_regions
                result["signature"]["confidence"] = 0.7  # Heuristic confidence
            
            if stamp_regions:
                result["stamp"]["present"] = True
                result["stamp"]["detections"] = stamp_regions
                result["stamp"]["confidence"] = 0.7
                
        except Exception as e:
            logger.error(f"Heuristic detection failed: {e}")
        
        return result
    
    def _detect_signature_regions(self, gray_region: np.ndarray, y_offset: int) -> List[Dict]:
        """Detect potential signature regions using edge detection"""
        detections = []
        
        try:
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = gray_region.shape
            min_area = width * height * 0.005  # Minimum 0.5% of region
            max_area = width * height * 0.15   # Maximum 15% of region
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Signatures typically have aspect ratio between 1.5 and 6
                    if 1.5 < aspect_ratio < 6:
                        detections.append({
                            "bbox": {
                                "x1": x,
                                "y1": y + y_offset,
                                "x2": x + w,
                                "y2": y + h + y_offset
                            },
                            "confidence": 0.65
                        })
            
            # Merge overlapping detections
            detections = self._merge_overlapping_boxes(detections)
            
        except Exception as e:
            logger.error(f"Signature region detection failed: {e}")
        
        return detections[:3]  # Return top 3 candidates
    
    def _detect_stamp_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect potential stamp regions using color and shape analysis"""
        detections = []
        
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect common stamp colors (blue, red, purple)
            color_ranges = [
                # Blue stamps
                (np.array([100, 50, 50]), np.array([130, 255, 255])),
                # Red stamps
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255])),
                # Purple stamps
                (np.array([130, 50, 50]), np.array([160, 255, 255])),
            ]
            
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = image.shape[:2]
            min_area = width * height * 0.005
            max_area = width * height * 0.1
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check for circular or square shape (stamps are usually round/square)
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2 + 1e-6)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if circularity > 0.4 or 0.7 < aspect_ratio < 1.4:
                        detections.append({
                            "bbox": {
                                "x1": x,
                                "y1": y,
                                "x2": x + w,
                                "y2": y + h
                            },
                            "confidence": 0.6 + circularity * 0.3
                        })
            
        except Exception as e:
            logger.error(f"Stamp region detection failed: {e}")
        
        return detections[:5]  # Return top 5 candidates
    
    def _merge_overlapping_boxes(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping bounding boxes"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        merged = []
        used = [False] * len(detections)
        
        for i, det1 in enumerate(detections):
            if used[i]:
                continue
            
            merged_box = det1.copy()
            used[i] = True
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if used[j]:
                    continue
                
                iou = self._calculate_iou(det1["bbox"], det2["bbox"])
                
                if iou > iou_threshold:
                    used[j] = True
                    # Expand bounding box to include both
                    merged_box["bbox"]["x1"] = min(merged_box["bbox"]["x1"], det2["bbox"]["x1"])
                    merged_box["bbox"]["y1"] = min(merged_box["bbox"]["y1"], det2["bbox"]["y1"])
                    merged_box["bbox"]["x2"] = max(merged_box["bbox"]["x2"], det2["bbox"]["x2"])
                    merged_box["bbox"]["y2"] = max(merged_box["bbox"]["y2"], det2["bbox"]["y2"])
            
            merged.append(merged_box)
        
        return merged
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
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
    
    def _merge_detections(self, yolo_result: Dict, heuristic_result: Dict) -> Dict:
        """Merge YOLO and heuristic detection results"""
        merged = yolo_result.copy()
        
        for category in ["signature", "stamp"]:
            if not merged[category]["present"] and heuristic_result[category]["present"]:
                merged[category] = heuristic_result[category]
            elif merged[category]["present"] and heuristic_result[category]["present"]:
                # Combine detections
                merged[category]["detections"].extend(heuristic_result[category]["detections"])
                merged[category]["detections"] = self._merge_overlapping_boxes(
                    merged[category]["detections"]
                )
        
        return merged
    
    def visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw detection bounding boxes on image for visualization
        """
        vis_image = image.copy()
        
        # Colors for each category
        colors = {
            "signature": (0, 255, 0),   # Green
            "stamp": (255, 0, 0)         # Blue (BGR)
        }
        
        for category, data in detections.items():
            if category not in colors:
                continue
                
            color = colors[category]
            
            for det in data.get("detections", []):
                bbox = det["bbox"]
                conf = det.get("confidence", 0)
                
                # Draw rectangle
                cv2.rectangle(
                    vis_image,
                    (bbox["x1"], bbox["y1"]),
                    (bbox["x2"], bbox["y2"]),
                    color, 2
                )
                
                # Add label
                label = f"{category}: {conf:.2f}"
                cv2.putText(
                    vis_image, label,
                    (bbox["x1"], bbox["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        return vis_image
