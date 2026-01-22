"""
Agentic AI Service for Intelligent Document Processing
Provides self-improving extraction with feedback loops and reasoning
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class AgenticAI:
    """
    Agentic AI system that:
    1. Plans extraction strategies based on document characteristics
    2. Validates and verifies extracted data
    3. Self-corrects using multiple extraction methods
    4. Learns from feedback to improve future extractions
    5. Provides explainable reasoning for extractions
    """
    
    def __init__(self, 
                 openai_api_key: str = None,
                 google_api_key: str = None,
                 feedback_store_path: str = None):
        """
        Initialize Agentic AI
        
        Args:
            openai_api_key: OpenAI API key for GPT-4
            google_api_key: Google API key for Gemini
            feedback_store_path: Path to store feedback data
        """
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.feedback_store_path = feedback_store_path or "./data/feedback"
        
        self.llm_client = None
        self._initialize_llm()
        
        # Load historical feedback for learning
        self.feedback_history = self._load_feedback_history()
        
        # Extraction strategies
        self.strategies = [
            "pattern_based",
            "fuzzy_matching",
            "llm_vision",
            "llm_text",
            "ensemble"
        ]
        
        logger.info("AgenticAI initialized")
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            if self.openai_api_key:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=self.openai_api_key)
                self.llm_provider = "openai"
            elif self.google_api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                self.llm_client = genai
                self.llm_provider = "google"
            else:
                logger.warning("No API keys provided. Using rule-based agent only.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
    
    def _load_feedback_history(self) -> List[Dict]:
        """Load historical feedback for learning"""
        feedback_file = Path(self.feedback_store_path) / "feedback_history.json"
        
        if feedback_file.exists():
            try:
                with open(feedback_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load feedback history: {e}")
        
        return []
    
    def _save_feedback(self, feedback: Dict):
        """Save feedback to history"""
        self.feedback_history.append(feedback)
        
        feedback_dir = Path(self.feedback_store_path)
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        feedback_file = feedback_dir / "feedback_history.json"
        
        try:
            with open(feedback_file, "w") as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def plan_extraction(self, document_info: Dict) -> Dict:
        """
        Plan extraction strategy based on document characteristics
        
        Args:
            document_info: Information about the document (format, language, quality)
            
        Returns:
            Extraction plan with recommended strategies
        """
        plan = {
            "strategies": [],
            "priority_fields": [],
            "estimated_confidence": {},
            "reasoning": []
        }
        
        # Analyze document characteristics
        is_multilingual = document_info.get("detected_languages", ["en"]) != ["en"]
        is_handwritten = document_info.get("has_handwriting", False)
        image_quality = document_info.get("quality_score", 0.8)
        
        # Plan based on characteristics
        if is_multilingual:
            plan["strategies"].append("llm_vision")
            plan["reasoning"].append("Document contains multiple languages - using LLM vision for better multilingual support")
        
        if is_handwritten:
            plan["strategies"].append("llm_vision")
            plan["reasoning"].append("Handwritten content detected - prioritizing vision-based extraction")
        
        if image_quality < 0.6:
            plan["strategies"].append("pattern_based")
            plan["reasoning"].append("Low image quality - using pattern matching as fallback")
        
        # Default strategies
        if not plan["strategies"]:
            plan["strategies"] = ["pattern_based", "fuzzy_matching", "llm_text"]
            plan["reasoning"].append("Standard document - using default extraction pipeline")
        
        # Learn from historical feedback
        similar_docs = self._find_similar_documents(document_info)
        if similar_docs:
            best_strategy = self._get_best_strategy_from_history(similar_docs)
            if best_strategy and best_strategy not in plan["strategies"]:
                plan["strategies"].insert(0, best_strategy)
                plan["reasoning"].append(f"Historical data suggests {best_strategy} works best for similar documents")
        
        plan["priority_fields"] = ["horse_power", "asset_cost", "model_name", "dealer_name"]
        
        return plan
    
    def validate_extraction(self, extracted: Dict, ocr_text: str, 
                           image: np.ndarray = None) -> Dict:
        """
        Validate extracted fields using multiple checks
        
        Args:
            extracted: Extracted field values
            ocr_text: Raw OCR text for verification
            image: Original image for visual verification
            
        Returns:
            Validation results with corrections if needed
        """
        validation = {
            "is_valid": True,
            "field_validations": {},
            "corrections": {},
            "reasoning": []
        }
        
        # Validate each field
        for field_name, field_data in extracted.get("fields", {}).items():
            field_valid = True
            issues = []
            
            value = field_data.get("value")
            confidence = field_data.get("confidence", 0)
            
            # Check confidence threshold
            if confidence < 0.5:
                field_valid = False
                issues.append(f"Low confidence ({confidence:.2f})")
            
            # Field-specific validation
            if field_name == "horse_power":
                if value is not None:
                    if not (20 <= value <= 200):
                        field_valid = False
                        issues.append(f"HP value {value} outside valid range (20-200)")
                    elif not self._verify_in_text(str(value), ocr_text, ["HP", "hp", "H.P."]):
                        issues.append("HP value not found in OCR text")
            
            elif field_name == "asset_cost":
                if value is not None:
                    if not (100000 <= value <= 50000000):
                        field_valid = False
                        issues.append(f"Cost {value} outside valid range")
                    elif not self._verify_numeric_in_text(value, ocr_text):
                        issues.append("Cost not found in OCR text")
            
            elif field_name == "model_name":
                if value and len(str(value)) < 3:
                    field_valid = False
                    issues.append("Model name too short")
            
            elif field_name == "dealer_name":
                if value and len(str(value)) < 5:
                    field_valid = False
                    issues.append("Dealer name too short")
            
            validation["field_validations"][field_name] = {
                "is_valid": field_valid,
                "issues": issues,
                "original_value": value,
                "original_confidence": confidence
            }
            
            if not field_valid:
                validation["is_valid"] = False
        
        # If validation failed, attempt corrections
        if not validation["is_valid"]:
            corrections = self._attempt_corrections(
                extracted, 
                validation["field_validations"],
                ocr_text,
                image
            )
            validation["corrections"] = corrections
        
        return validation
    
    def _verify_in_text(self, value: str, text: str, context_keywords: List[str]) -> bool:
        """Verify if value exists in text near context keywords"""
        import re
        
        for keyword in context_keywords:
            pattern = rf'{keyword}\s*[-:]?\s*{re.escape(value)}|{re.escape(value)}\s*{keyword}'
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return value in text
    
    def _verify_numeric_in_text(self, value: float, text: str) -> bool:
        """Verify if numeric value exists in text"""
        # Format value in different ways
        formats = [
            str(int(value)),
            f"{int(value):,}",
            f"{value:.2f}",
            f"{value:,.2f}"
        ]
        
        for fmt in formats:
            if fmt in text or fmt.replace(",", "") in text:
                return True
        
        return False
    
    def _attempt_corrections(self, extracted: Dict, validations: Dict,
                            ocr_text: str, image: np.ndarray = None) -> Dict:
        """Attempt to correct invalid extractions"""
        corrections = {}
        
        for field_name, validation in validations.items():
            if not validation["is_valid"]:
                logger.info(f"Attempting correction for {field_name}")
                
                # Try re-extraction with different method
                corrected_value = self._re_extract_field(
                    field_name, 
                    ocr_text, 
                    validation["original_value"],
                    image
                )
                
                if corrected_value is not None:
                    corrections[field_name] = {
                        "original": validation["original_value"],
                        "corrected": corrected_value,
                        "method": "re-extraction"
                    }
        
        return corrections
    
    def _re_extract_field(self, field_name: str, ocr_text: str, 
                         original_value: Any, image: np.ndarray = None) -> Any:
        """Re-extract a specific field using alternative methods"""
        import re
        
        if field_name == "horse_power":
            # Try more aggressive HP extraction
            patterns = [
                r'(\d{2})\s*[Hh][Pp]',
                r'(\d{2})\s*H\.P',
                r'HP[-\s:]+(\d{2})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, ocr_text)
                if match:
                    hp = int(match.group(1))
                    if 20 <= hp <= 100:
                        return hp
        
        elif field_name == "asset_cost":
            # Look for largest number that could be a cost
            numbers = re.findall(r'[\d,]+(?:\.\d{2})?', ocr_text)
            valid_costs = []
            
            for num_str in numbers:
                try:
                    num = float(num_str.replace(",", ""))
                    if 100000 <= num <= 50000000:
                        valid_costs.append(num)
                except ValueError:
                    continue
            
            if valid_costs:
                return max(valid_costs)  # Return the largest valid cost
        
        return None
    
    def provide_explanation(self, result: Dict) -> Dict:
        """
        Provide human-readable explanation for extraction results
        
        Args:
            result: Extraction result
            
        Returns:
            Explanation with reasoning for each field
        """
        explanation = {
            "summary": "",
            "field_explanations": {},
            "confidence_assessment": "",
            "recommendations": []
        }
        
        fields = result.get("fields", {})
        
        # Generate field explanations
        for field_name, field_data in fields.items():
            value = field_data.get("value")
            confidence = field_data.get("confidence", 0)
            method = field_data.get("extraction_method", "unknown")
            
            if value is not None:
                explanation["field_explanations"][field_name] = {
                    "value": value,
                    "reasoning": f"Extracted using {method} method with {confidence:.0%} confidence",
                    "confidence_level": self._get_confidence_level(confidence)
                }
            else:
                explanation["field_explanations"][field_name] = {
                    "value": None,
                    "reasoning": "Could not extract this field",
                    "confidence_level": "none"
                }
        
        # Overall summary
        high_conf_count = sum(
            1 for f in fields.values() 
            if f.get("confidence", 0) >= 0.8
        )
        total_fields = len(fields)
        
        explanation["summary"] = (
            f"Extracted {high_conf_count}/{total_fields} fields with high confidence. "
            f"Overall confidence: {result.get('metadata', {}).get('overall_confidence', 0):.0%}"
        )
        
        # Recommendations
        low_conf_fields = [
            name for name, data in fields.items() 
            if data.get("confidence", 0) < 0.7
        ]
        
        if low_conf_fields:
            explanation["recommendations"].append(
                f"Manual review recommended for: {', '.join(low_conf_fields)}"
            )
        
        return explanation
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def process_feedback(self, document_id: str, field_name: str,
                        predicted_value: Any, correct_value: Any,
                        extraction_method: str):
        """
        Process user feedback for learning
        
        Args:
            document_id: ID of the document
            field_name: Name of the field
            predicted_value: What was predicted
            correct_value: What was correct
            extraction_method: Method used for extraction
        """
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "document_id": document_id,
            "field_name": field_name,
            "predicted_value": predicted_value,
            "correct_value": correct_value,
            "extraction_method": extraction_method,
            "was_correct": predicted_value == correct_value
        }
        
        self._save_feedback(feedback)
        
        logger.info(f"Feedback recorded for {field_name}: {predicted_value} -> {correct_value}")
    
    def _find_similar_documents(self, document_info: Dict) -> List[Dict]:
        """Find similar documents from feedback history"""
        similar = []
        
        doc_type = document_info.get("document_type", "invoice")
        languages = set(document_info.get("detected_languages", ["en"]))
        
        for feedback in self.feedback_history:
            fb_type = feedback.get("document_type", "invoice")
            fb_langs = set(feedback.get("detected_languages", ["en"]))
            
            if fb_type == doc_type and fb_langs.intersection(languages):
                similar.append(feedback)
        
        return similar[-10:]  # Return last 10 similar
    
    def _get_best_strategy_from_history(self, similar_docs: List[Dict]) -> Optional[str]:
        """Determine best extraction strategy from historical data"""
        if not similar_docs:
            return None
        
        strategy_scores = {}
        
        for doc in similar_docs:
            method = doc.get("extraction_method", "pattern_based")
            was_correct = doc.get("was_correct", False)
            
            if method not in strategy_scores:
                strategy_scores[method] = {"correct": 0, "total": 0}
            
            strategy_scores[method]["total"] += 1
            if was_correct:
                strategy_scores[method]["correct"] += 1
        
        best_strategy = None
        best_score = 0
        
        for strategy, scores in strategy_scores.items():
            if scores["total"] > 0:
                accuracy = scores["correct"] / scores["total"]
                if accuracy > best_score:
                    best_score = accuracy
                    best_strategy = strategy
        
        return best_strategy
    
    async def intelligent_extract(self, ocr_text: str, image: np.ndarray,
                                  document_info: Dict = None) -> Dict:
        """
        Intelligent extraction using agentic reasoning
        
        This method:
        1. Analyzes document and plans extraction
        2. Runs multiple extraction strategies
        3. Validates and cross-checks results
        4. Provides explainable output
        """
        document_info = document_info or {}
        
        # Step 1: Plan extraction
        plan = self.plan_extraction(document_info)
        
        # Step 2: Execute extraction strategies
        results = {}
        for strategy in plan["strategies"]:
            if strategy == "pattern_based":
                from .field_extractor import FieldExtractor
                extractor = FieldExtractor(use_llm=False)
                results["pattern_based"] = extractor._extract_with_patterns(ocr_text)
            
            elif strategy == "fuzzy_matching":
                from .field_extractor import FieldExtractor
                extractor = FieldExtractor(use_llm=False)
                results["fuzzy_matching"] = extractor._extract_with_fuzzy_matching(ocr_text)
            
            elif strategy == "llm_vision" and self.llm_client:
                results["llm_vision"] = await self._llm_extraction(ocr_text, image, use_vision=True)
            
            elif strategy == "llm_text" and self.llm_client:
                results["llm_text"] = await self._llm_extraction(ocr_text, image, use_vision=False)
        
        # Step 3: Ensemble results
        final_result = self._ensemble_results(results)
        
        # Step 4: Validate
        validation = self.validate_extraction({"fields": final_result}, ocr_text, image)
        
        # Step 5: Apply corrections
        if validation["corrections"]:
            for field, correction in validation["corrections"].items():
                if field in final_result:
                    final_result[field]["value"] = correction["corrected"]
                    final_result[field]["was_corrected"] = True
        
        # Step 6: Generate explanation
        explanation = self.provide_explanation({"fields": final_result})
        
        return {
            "fields": final_result,
            "plan": plan,
            "validation": validation,
            "explanation": explanation
        }
    
    async def _llm_extraction(self, ocr_text: str, image: np.ndarray,
                             use_vision: bool = True) -> Dict:
        """Extract fields using LLM"""
        # Implementation would call LLM API
        # This is a placeholder that returns empty results
        return {
            "dealer_name": {"value": None, "confidence": 0.0, "method": "llm"},
            "model_name": {"value": None, "confidence": 0.0, "method": "llm"},
            "horse_power": {"value": None, "confidence": 0.0, "method": "llm"},
            "asset_cost": {"value": None, "confidence": 0.0, "method": "llm"},
        }
    
    def _ensemble_results(self, results: Dict[str, Dict]) -> Dict:
        """Combine results from multiple strategies"""
        final = {
            "dealer_name": {"value": None, "confidence": 0.0, "method": None},
            "model_name": {"value": None, "confidence": 0.0, "method": None},
            "horse_power": {"value": None, "confidence": 0.0, "method": None},
            "asset_cost": {"value": None, "confidence": 0.0, "method": None},
        }
        
        for field in final.keys():
            best_confidence = 0
            best_value = None
            best_method = None
            
            for strategy, result in results.items():
                if field in result:
                    confidence = result[field].get("confidence", 0)
                    value = result[field].get("value")
                    
                    if confidence > best_confidence and value is not None:
                        best_confidence = confidence
                        best_value = value
                        best_method = strategy
            
            if best_value is not None:
                final[field] = {
                    "value": best_value,
                    "confidence": best_confidence,
                    "method": best_method
                }
        
        return final
