"""
Agentic AI Orchestration Layer
Multi-agent system for intelligent document processing
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Roles for specialized agents"""
    COORDINATOR = "coordinator"
    OCR_SPECIALIST = "ocr_specialist"
    VISION_ANALYST = "vision_analyst"
    FIELD_EXTRACTOR = "field_extractor"
    VALIDATOR = "validator"
    REASONER = "reasoner"

class TaskStatus(Enum):
    """Status of agent tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

@dataclass
class AgentMessage:
    """Message passed between agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: AgentRole = AgentRole.COORDINATOR
    recipient: AgentRole = AgentRole.COORDINATOR
    content: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1

@dataclass
class AgentTask:
    """Task assigned to an agent"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    assigned_to: AgentRole = AgentRole.COORDINATOR
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None

@dataclass
class ReasoningStep:
    """A step in the reasoning chain"""
    step_num: int
    thought: str
    action: str
    observation: str
    confidence: float = 0.0

class Tool(ABC):
    """Base class for agent tools"""
    name: str = "base_tool"
    description: str = "Base tool"
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict:
        pass

class OCRTool(Tool):
    """Tool for OCR extraction"""
    name = "ocr_extract"
    description = "Extract text from document image using OCR"
    
    def __init__(self, ocr_service):
        self.ocr_service = ocr_service
    
    async def execute(self, image=None, **kwargs) -> Dict:
        if image is None:
            return {"error": "No image provided"}
        result = self.ocr_service.extract_structured_text(image)
        return {"text": result["raw_text"], "blocks": result["text_blocks"], "confidence": result["confidence"]}

class VisionTool(Tool):
    """Tool for vision analysis"""
    name = "vision_detect"
    description = "Detect signatures and stamps in document"
    
    def __init__(self, vision_service):
        self.vision_service = vision_service
    
    async def execute(self, image=None, **kwargs) -> Dict:
        if image is None:
            return {"error": "No image provided"}
        return self.vision_service.detect_signatures_stamps(image)

class PatternMatchTool(Tool):
    """Tool for pattern-based extraction"""
    name = "pattern_extract"
    description = "Extract fields using regex patterns"
    
    def __init__(self, field_extractor):
        self.extractor = field_extractor
    
    async def execute(self, text: str = "", **kwargs) -> Dict:
        return self.extractor._extract_with_patterns(text)

class FuzzyMatchTool(Tool):
    """Tool for fuzzy matching against master data"""
    name = "fuzzy_match"
    description = "Match extracted values against master lists"
    
    def __init__(self, field_extractor):
        self.extractor = field_extractor
    
    async def execute(self, text: str = "", **kwargs) -> Dict:
        return self.extractor._extract_with_fuzzy_matching(text)

class ValidateTool(Tool):
    """Tool for validating extracted data"""
    name = "validate"
    description = "Validate extracted fields for consistency"
    
    async def execute(self, fields: Dict = None, text: str = "", **kwargs) -> Dict:
        validation = {"is_valid": True, "issues": []}
        if not fields:
            return validation
        
        # Validate horse power
        hp = fields.get("horse_power", {}).get("value")
        if hp and not (20 <= hp <= 200):
            validation["is_valid"] = False
            validation["issues"].append(f"HP {hp} outside valid range")
        
        # Validate cost
        cost = fields.get("asset_cost", {}).get("value")
        if cost and not (100000 <= cost <= 50000000):
            validation["is_valid"] = False
            validation["issues"].append(f"Cost {cost} outside valid range")
        
        return validation

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, role: AgentRole, tools: List[Tool] = None):
        self.role = role
        self.tools = {t.name: t for t in (tools or [])}
        self.memory: List[Dict] = []
        self.reasoning_chain: List[ReasoningStep] = []
    
    def add_to_memory(self, item: Dict):
        self.memory.append({"timestamp": datetime.now().isoformat(), **item})
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]
    
    def add_reasoning_step(self, thought: str, action: str, observation: str, confidence: float = 0.0):
        step = ReasoningStep(
            step_num=len(self.reasoning_chain) + 1,
            thought=thought, action=action, observation=observation, confidence=confidence
        )
        self.reasoning_chain.append(step)
        return step
    
    async def use_tool(self, tool_name: str, **kwargs) -> Dict:
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not available"}
        return await self.tools[tool_name].execute(**kwargs)
    
    @abstractmethod
    async def process(self, task: AgentTask) -> AgentTask:
        pass
    
    def get_reasoning_summary(self) -> str:
        return "\n".join([
            f"Step {s.step_num}: {s.thought} -> {s.action} -> {s.observation}"
            for s in self.reasoning_chain
        ])

class OCRAgent(BaseAgent):
    """Agent specialized in OCR extraction"""
    
    def __init__(self, ocr_service):
        tools = [OCRTool(ocr_service)]
        super().__init__(AgentRole.OCR_SPECIALIST, tools)
    
    async def process(self, task: AgentTask) -> AgentTask:
        task.status = TaskStatus.IN_PROGRESS
        image = task.input_data.get("image")
        
        self.add_reasoning_step(
            thought="Need to extract text from document image",
            action="Using OCR tool to extract text",
            observation="Processing..."
        )
        
        result = await self.use_tool("ocr_extract", image=image)
        
        if "error" in result:
            task.status = TaskStatus.FAILED
            task.error = result["error"]
        else:
            self.add_reasoning_step(
                thought=f"OCR completed with {result['confidence']:.0%} confidence",
                action="Returning extracted text",
                observation=f"Extracted {len(result['blocks'])} text blocks",
                confidence=result["confidence"]
            )
            task.output_data = result
            task.status = TaskStatus.COMPLETED
        
        task.completed_at = datetime.now().isoformat()
        return task

class VisionAgent(BaseAgent):
    """Agent specialized in visual analysis"""
    
    def __init__(self, vision_service):
        tools = [VisionTool(vision_service)]
        super().__init__(AgentRole.VISION_ANALYST, tools)
    
    async def process(self, task: AgentTask) -> AgentTask:
        task.status = TaskStatus.IN_PROGRESS
        image = task.input_data.get("image")
        
        self.add_reasoning_step(
            thought="Need to detect signatures and stamps",
            action="Using vision detection tool",
            observation="Scanning document..."
        )
        
        result = await self.use_tool("vision_detect", image=image)
        
        sig_found = result.get("signature", {}).get("present", False)
        stamp_found = result.get("stamp", {}).get("present", False)
        
        self.add_reasoning_step(
            thought=f"Detection complete: Signature={'Yes' if sig_found else 'No'}, Stamp={'Yes' if stamp_found else 'No'}",
            action="Returning detection results",
            observation=f"Found {len(result.get('signature', {}).get('detections', []))} signatures, {len(result.get('stamp', {}).get('detections', []))} stamps",
            confidence=max(result.get("signature", {}).get("confidence", 0), result.get("stamp", {}).get("confidence", 0))
        )
        
        task.output_data = result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().isoformat()
        return task

class ExtractorAgent(BaseAgent):
    """Agent specialized in field extraction"""
    
    def __init__(self, field_extractor):
        tools = [PatternMatchTool(field_extractor), FuzzyMatchTool(field_extractor)]
        super().__init__(AgentRole.FIELD_EXTRACTOR, tools)
        self.extractor = field_extractor
    
    async def process(self, task: AgentTask) -> AgentTask:
        task.status = TaskStatus.IN_PROGRESS
        text = task.input_data.get("text", "")
        
        # Strategy 1: Pattern matching
        self.add_reasoning_step(
            thought="Starting with pattern-based extraction",
            action="Applying regex patterns for HP, cost, model, dealer",
            observation="Processing..."
        )
        pattern_result = await self.use_tool("pattern_extract", text=text)
        
        # Strategy 2: Fuzzy matching
        self.add_reasoning_step(
            thought="Enhancing with fuzzy matching against master data",
            action="Matching dealer and model names",
            observation="Processing..."
        )
        fuzzy_result = await self.use_tool("fuzzy_match", text=text)
        
        # Combine results
        fields = {}
        for field in ["dealer_name", "model_name", "horse_power", "asset_cost"]:
            pattern_val = pattern_result.get(field, {})
            fuzzy_val = fuzzy_result.get(field, {}) if field in ["dealer_name", "model_name"] else {}
            
            if fuzzy_val.get("confidence", 0) > pattern_val.get("confidence", 0):
                fields[field] = fuzzy_val
            else:
                fields[field] = pattern_val
        
        self.add_reasoning_step(
            thought="Combined pattern and fuzzy matching results",
            action="Selected highest confidence values",
            observation=f"Extracted: HP={fields.get('horse_power', {}).get('value')}, Cost={fields.get('asset_cost', {}).get('value')}",
            confidence=sum(f.get("confidence", 0) for f in fields.values()) / len(fields)
        )
        
        task.output_data = {"fields": fields}
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().isoformat()
        return task

class ValidatorAgent(BaseAgent):
    """Agent specialized in validation"""
    
    def __init__(self):
        tools = [ValidateTool()]
        super().__init__(AgentRole.VALIDATOR, tools)
    
    async def process(self, task: AgentTask) -> AgentTask:
        task.status = TaskStatus.IN_PROGRESS
        fields = task.input_data.get("fields", {})
        text = task.input_data.get("text", "")
        
        self.add_reasoning_step(
            thought="Validating extracted fields",
            action="Checking value ranges and consistency",
            observation="Processing..."
        )
        
        result = await self.use_tool("validate", fields=fields, text=text)
        
        if result["is_valid"]:
            self.add_reasoning_step(
                thought="All validations passed",
                action="Approving extraction",
                observation="Fields are valid",
                confidence=1.0
            )
            task.status = TaskStatus.COMPLETED
        else:
            self.add_reasoning_step(
                thought=f"Validation issues: {result['issues']}",
                action="Flagging for review",
                observation="Some fields need correction",
                confidence=0.5
            )
            task.status = TaskStatus.NEEDS_REVIEW
        
        task.output_data = result
        task.completed_at = datetime.now().isoformat()
        return task
