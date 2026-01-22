"""
Field Extraction Service using LLM and rule-based methods
Extracts structured fields from OCR text and vision analysis
"""
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz
import numpy as np

logger = logging.getLogger(__name__)

class FieldExtractor:
    """
    Extracts structured fields from invoices using a combination of:
    1. Pattern matching (regex)
    2. Fuzzy matching for dealer/model lookup
    3. LLM-based extraction for complex cases
    """
    
    # Common patterns for field extraction
    PATTERNS = {
        "horse_power": [
            r'(\d{2,3})\s*[Hh][Pp]',
            r'(\d{2,3})\s*H\.?P\.?',
            r'HP\s*[-:]?\s*(\d{2,3})',
            r'Horse\s*Power\s*[-:]?\s*(\d{2,3})',
            r'(\d{2,3})\s*एचपी',  # Hindi
            r'(\d{2,3})\s*अश्वशक्ति',
        ],
        "cost": [
            r'(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d{2})?)',
            r'Total\s*[-:]?\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'Grand\s*Total\s*[-:]?\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'Amount\s*[-:]?\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'Full\s*Cost.*?(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'([\d,]+(?:\.\d{2})?)\s*(?:only|Only)',
            r'Cost\s*of\s*Tractor\s*[-:]?\s*(?:Rs\.?|₹)?\s*([\d,]+(?:\.\d{2})?)',
        ],
        "model": [
            r'Model\s*[-:]?\s*([A-Za-z0-9\s\-+]+(?:\d+[A-Za-z]*|\s*[IVX]+)?)',
            r'Tractor\s+([A-Z]{2,}\s*[-]?\s*\d{2,4}[A-Za-z\s\-+]*)',
            r'([A-Z]{2,}\s*[-]?\s*\d{3,4}\s*[A-Za-z\-+]*)',
            r'(SWARAJ|SONALIKA|MAHINDRA|JOHN DEERE|NEW HOLLAND|KUBOTA|MF|MASSEY|TAFE)[\s\-]*([A-Za-z0-9\s\-+]+)',
        ],
        "dealer": [
            r'^([A-Z][A-Za-z\s\.]+(?:Ltd\.?|Limited|Corporation|Tractors|Sales|Dealers?))',
            r'Dealer\s*[-:]?\s*([A-Za-z\s\.]+)',
            r'([A-Z][A-Za-z\s\.]+Tractors)',
            r'([A-Z][A-Za-z\s\.]+(?:Agro|Agriculture|Industries)[\s]+[A-Za-z\.]+)',
        ]
    }
    
    # Known dealer master list (would be loaded from database in production)
    DEALER_MASTER = [
        "The Odisha Agro Industries Corporation Ltd",
        "International Tractors Ltd",
        "Sri Amutham Tractors",
        "Mahindra & Mahindra Ltd",
        "TAFE Motors and Tractors Ltd",
        "National Tractor Sales",
        "Kubota Tractor Corporation",
        "John Deere India",
        "New Holland Agriculture",
        "Escorts Kubota Ltd",
        "Sonalika International Tractors Ltd",
        "नेशनल ट्रैक्टर सेल्स",  # Hindi
    ]
    
    # Known model master list
    MODEL_MASTER = [
        "DI-745 III HDM+4WD",
        "SWARAJ 744 FE",
        "SONALIKA TIGER 55-4WD",
        "MF 241 DI",
        "MF 1035 DI",
        "MF 7250 DI",
        "MF 9000 DI",
        "TAFE 9500",
        "Kubota MU 5502 4WD",
        "Mahindra 475 DI",
        "John Deere 5050D",
        "New Holland 3630",
    ]
    
    def __init__(self, use_llm: bool = True, openai_api_key: str = None, google_api_key: str = None):
        self.use_llm = use_llm
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.llm_client = None
        
        if use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client (OpenAI or Google)"""
        try:
            if self.openai_api_key:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=self.openai_api_key)
                self.llm_provider = "openai"
                logger.info("OpenAI client initialized")
            elif self.google_api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                self.llm_client = genai
                self.llm_provider = "google"
                logger.info("Google Gemini client initialized")
            else:
                logger.warning("No API keys provided. LLM extraction disabled.")
                self.use_llm = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.use_llm = False
    
    def extract_fields(self, ocr_text: str, text_blocks: List[Dict] = None, 
                      image: np.ndarray = None) -> Dict:
        """
        Extract all required fields from OCR text
        
        Args:
            ocr_text: Raw OCR text
            text_blocks: Structured text blocks with bounding boxes
            image: Original image for LLM vision analysis
            
        Returns:
            Dict with extracted fields and confidence scores
        """
        result = {
            "dealer_name": {"value": None, "confidence": 0.0, "method": None},
            "model_name": {"value": None, "confidence": 0.0, "method": None},
            "horse_power": {"value": None, "confidence": 0.0, "method": None},
            "asset_cost": {"value": None, "confidence": 0.0, "method": None},
        }
        
        # Step 1: Pattern-based extraction
        pattern_results = self._extract_with_patterns(ocr_text)
        for field, value in pattern_results.items():
            if value["value"] is not None:
                result[field] = value
        
        # Step 2: Fuzzy matching for dealer and model
        fuzzy_results = self._extract_with_fuzzy_matching(ocr_text)
        for field in ["dealer_name", "model_name"]:
            if fuzzy_results[field]["confidence"] > result[field]["confidence"]:
                result[field] = fuzzy_results[field]
        
        # Step 3: LLM extraction for missing or low-confidence fields
        if self.use_llm and image is not None:
            llm_results = self._extract_with_llm(ocr_text, image)
            for field, value in llm_results.items():
                if field in result:
                    if result[field]["value"] is None or result[field]["confidence"] < 0.5:
                        if value["value"] is not None:
                            result[field] = value
        
        # Post-process and validate
        result = self._post_process_results(result)
        
        return result
    
    def _extract_with_patterns(self, text: str) -> Dict:
        """Extract fields using regex patterns"""
        result = {
            "dealer_name": {"value": None, "confidence": 0.0, "method": "pattern"},
            "model_name": {"value": None, "confidence": 0.0, "method": "pattern"},
            "horse_power": {"value": None, "confidence": 0.0, "method": "pattern"},
            "asset_cost": {"value": None, "confidence": 0.0, "method": "pattern"},
        }
        
        # Extract horse power
        for pattern in self.PATTERNS["horse_power"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    hp_value = int(match.group(1))
                    if 20 <= hp_value <= 200:  # Valid HP range
                        result["horse_power"]["value"] = hp_value
                        result["horse_power"]["confidence"] = 0.9
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract cost
        cost_matches = []
        for pattern in self.PATTERNS["cost"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    cost_str = match.replace(",", "").replace(" ", "")
                    cost_value = float(cost_str)
                    if cost_value > 10000:  # Reasonable minimum cost
                        cost_matches.append(cost_value)
                except (ValueError, TypeError):
                    continue
        
        if cost_matches:
            # Take the largest cost (usually the total)
            result["asset_cost"]["value"] = max(cost_matches)
            result["asset_cost"]["confidence"] = 0.85
        
        # Extract model
        for pattern in self.PATTERNS["model"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                model = match.group(1) if len(match.groups()) == 1 else " ".join(match.groups())
                model = model.strip()
                if len(model) > 3:
                    result["model_name"]["value"] = model
                    result["model_name"]["confidence"] = 0.75
                    break
        
        # Extract dealer (usually in header)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            for pattern in self.PATTERNS["dealer"]:
                match = re.search(pattern, line.strip())
                if match:
                    dealer = match.group(1).strip()
                    if len(dealer) > 5:
                        result["dealer_name"]["value"] = dealer
                        result["dealer_name"]["confidence"] = 0.7
                        break
            if result["dealer_name"]["value"]:
                break
        
        return result
    
    def _extract_with_fuzzy_matching(self, text: str) -> Dict:
        """Extract dealer and model using fuzzy matching against master lists"""
        result = {
            "dealer_name": {"value": None, "confidence": 0.0, "method": "fuzzy"},
            "model_name": {"value": None, "confidence": 0.0, "method": "fuzzy"},
        }
        
        # Match dealer
        best_dealer_match = None
        best_dealer_score = 0
        
        text_upper = text.upper()
        
        for dealer in self.DEALER_MASTER:
            score = fuzz.partial_ratio(dealer.upper(), text_upper)
            if score > best_dealer_score and score >= 70:
                best_dealer_score = score
                best_dealer_match = dealer
        
        if best_dealer_match:
            result["dealer_name"]["value"] = best_dealer_match
            result["dealer_name"]["confidence"] = best_dealer_score / 100.0
        
        # Match model
        best_model_match = None
        best_model_score = 0
        
        for model in self.MODEL_MASTER:
            score = fuzz.partial_ratio(model.upper(), text_upper)
            if score > best_model_score and score >= 75:
                best_model_score = score
                best_model_match = model
        
        if best_model_match:
            result["model_name"]["value"] = best_model_match
            result["model_name"]["confidence"] = best_model_score / 100.0
        
        return result
    
    def _extract_with_llm(self, text: str, image: np.ndarray) -> Dict:
        """Extract fields using LLM with vision capabilities"""
        result = {
            "dealer_name": {"value": None, "confidence": 0.0, "method": "llm"},
            "model_name": {"value": None, "confidence": 0.0, "method": "llm"},
            "horse_power": {"value": None, "confidence": 0.0, "method": "llm"},
            "asset_cost": {"value": None, "confidence": 0.0, "method": "llm"},
        }
        
        prompt = """Analyze this invoice/quotation document and extract the following fields.
Return ONLY a JSON object with these exact keys:

{
    "dealer_name": "Name of the dealer/company issuing the invoice",
    "model_name": "Tractor or equipment model name/number",
    "horse_power": <numeric value only, e.g., 50>,
    "asset_cost": <numeric value only, no currency symbols, e.g., 911769>
}

OCR Text from document:
"""
        prompt += text[:3000]  # Limit text length
        
        try:
            if self.llm_provider == "openai":
                response = self._call_openai(prompt, image)
            elif self.llm_provider == "google":
                response = self._call_google(prompt, image)
            else:
                return result
            
            # Parse LLM response
            if response:
                parsed = self._parse_llm_response(response)
                for field, value in parsed.items():
                    if value is not None:
                        result[field]["value"] = value
                        result[field]["confidence"] = 0.85
                        
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
        
        return result
    
    def _call_openai(self, prompt: str, image: np.ndarray = None) -> str:
        """Call OpenAI API"""
        import base64
        from io import BytesIO
        from PIL import Image
        
        messages = [{"role": "user", "content": []}]
        
        # Add text
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        # Add image if available
        if image is not None:
            # Convert numpy array to base64
            pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def _call_google(self, prompt: str, image: np.ndarray = None) -> str:
        """Call Google Gemini API"""
        from PIL import Image
        
        model = self.llm_client.GenerativeModel('gemini-pro-vision')
        
        content = [prompt]
        
        if image is not None:
            pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            content.append(pil_image)
        
        response = model.generate_content(content)
        
        return response.text
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""
        result = {}
        
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                if "dealer_name" in parsed:
                    result["dealer_name"] = parsed["dealer_name"]
                if "model_name" in parsed:
                    result["model_name"] = parsed["model_name"]
                if "horse_power" in parsed:
                    try:
                        result["horse_power"] = int(parsed["horse_power"])
                    except (ValueError, TypeError):
                        pass
                if "asset_cost" in parsed:
                    try:
                        cost = str(parsed["asset_cost"]).replace(",", "")
                        result["asset_cost"] = float(cost)
                    except (ValueError, TypeError):
                        pass
                        
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
        
        return result
    
    def _post_process_results(self, result: Dict) -> Dict:
        """Post-process and validate extracted results"""
        
        # Validate horse power
        if result["horse_power"]["value"] is not None:
            hp = result["horse_power"]["value"]
            if not (20 <= hp <= 200):
                result["horse_power"]["value"] = None
                result["horse_power"]["confidence"] = 0.0
        
        # Validate cost (should be reasonable for tractor)
        if result["asset_cost"]["value"] is not None:
            cost = result["asset_cost"]["value"]
            if not (100000 <= cost <= 50000000):  # 1 lakh to 5 crore
                result["asset_cost"]["value"] = None
                result["asset_cost"]["confidence"] = 0.0
        
        # Clean dealer name
        if result["dealer_name"]["value"]:
            result["dealer_name"]["value"] = result["dealer_name"]["value"].strip()
        
        # Clean model name
        if result["model_name"]["value"]:
            result["model_name"]["value"] = result["model_name"]["value"].strip()
        
        return result
    
    def validate_against_master(self, extracted: Dict) -> Dict:
        """Validate extracted values against master data"""
        validation = {
            "dealer_name": {"valid": False, "match_score": 0},
            "model_name": {"valid": False, "match_score": 0},
        }
        
        # Validate dealer
        if extracted["dealer_name"]["value"]:
            for dealer in self.DEALER_MASTER:
                score = fuzz.ratio(
                    extracted["dealer_name"]["value"].upper(),
                    dealer.upper()
                )
                if score >= 90:
                    validation["dealer_name"]["valid"] = True
                    validation["dealer_name"]["match_score"] = score
                    validation["dealer_name"]["matched_to"] = dealer
                    break
        
        # Validate model
        if extracted["model_name"]["value"]:
            for model in self.MODEL_MASTER:
                score = fuzz.ratio(
                    extracted["model_name"]["value"].upper(),
                    model.upper()
                )
                if score >= 85:
                    validation["model_name"]["valid"] = True
                    validation["model_name"]["match_score"] = score
                    validation["model_name"]["matched_to"] = model
                    break
        
        return validation
