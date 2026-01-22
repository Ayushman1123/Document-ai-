"""
Agentic AI Coordinator - Orchestrates multi-agent document processing
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

from .orchestration import (
    AgentRole, TaskStatus, AgentTask, AgentMessage,
    BaseAgent, OCRAgent, VisionAgent, ExtractorAgent, ValidatorAgent,
    ReasoningStep
)

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """State of the orchestration workflow"""
    document_id: str
    current_phase: str = "initialization"
    tasks: Dict[str, AgentTask] = field(default_factory=dict)
    messages: List[AgentMessage] = field(default_factory=list)
    reasoning_log: List[Dict] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    final_result: Optional[Dict] = None

class AgenticCoordinator:
    """
    Main orchestrator that coordinates multiple specialized agents
    
    Workflow:
    1. Plan: Analyze document and create execution plan
    2. Execute: Run agents in parallel/sequence as needed
    3. Validate: Cross-check results between agents
    4. Refine: Self-correct using feedback loops
    5. Synthesize: Combine all results into final output
    """
    
    def __init__(self, ocr_service=None, vision_service=None, field_extractor=None):
        self.agents: Dict[AgentRole, BaseAgent] = {}
        
        if ocr_service:
            self.agents[AgentRole.OCR_SPECIALIST] = OCRAgent(ocr_service)
        if vision_service:
            self.agents[AgentRole.VISION_ANALYST] = VisionAgent(vision_service)
        if field_extractor:
            self.agents[AgentRole.FIELD_EXTRACTOR] = ExtractorAgent(field_extractor)
        
        self.agents[AgentRole.VALIDATOR] = ValidatorAgent()
        self.workflow_states: Dict[str, WorkflowState] = {}
        
        logger.info(f"AgenticCoordinator initialized with {len(self.agents)} agents")
    
    def log_reasoning(self, state: WorkflowState, agent: str, thought: str, action: str, result: str):
        """Log reasoning step for explainability"""
        state.reasoning_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "thought": thought,
            "action": action,
            "result": result
        })
    
    async def process_document(self, document_id: str, image: np.ndarray) -> Dict:
        """
        Main entry point for processing a document with agentic orchestration
        """
        state = WorkflowState(document_id=document_id)
        self.workflow_states[document_id] = state
        
        try:
            # Phase 1: Planning
            state.current_phase = "planning"
            self.log_reasoning(state, "Coordinator", 
                "New document received. Need to plan extraction strategy.",
                "Analyzing document characteristics",
                "Creating parallel execution plan for OCR and Vision")
            
            execution_plan = self._create_execution_plan(image)
            
            # Phase 2: Parallel Execution of OCR and Vision
            state.current_phase = "extraction"
            self.log_reasoning(state, "Coordinator",
                "Executing OCR and Vision analysis in parallel",
                "Dispatching tasks to specialized agents",
                "Waiting for results...")
            
            ocr_task, vision_task = await self._execute_parallel([
                self._create_ocr_task(document_id, image),
                self._create_vision_task(document_id, image)
            ])
            
            state.tasks["ocr"] = ocr_task
            state.tasks["vision"] = vision_task
            
            # Check OCR success
            if ocr_task.status != TaskStatus.COMPLETED:
                raise Exception(f"OCR failed: {ocr_task.error}")
            
            ocr_text = ocr_task.output_data.get("text", "")
            
            self.log_reasoning(state, "OCR Agent",
                f"Extracted {len(ocr_task.output_data.get('blocks', []))} text blocks",
                "Completed text extraction",
                f"Confidence: {ocr_task.output_data.get('confidence', 0):.0%}")
            
            self.log_reasoning(state, "Vision Agent",
                f"Scanned for signatures and stamps",
                "Completed visual detection",
                f"Signature: {'Found' if vision_task.output_data.get('signature', {}).get('present') else 'Not found'}, "
                f"Stamp: {'Found' if vision_task.output_data.get('stamp', {}).get('present') else 'Not found'}")
            
            # Phase 3: Field Extraction
            state.current_phase = "field_extraction"
            self.log_reasoning(state, "Coordinator",
                "OCR complete. Now extracting structured fields.",
                "Dispatching to Field Extractor agent",
                "Using pattern matching + fuzzy matching")
            
            extract_task = await self._execute_extraction(document_id, ocr_text, image)
            state.tasks["extraction"] = extract_task
            
            if extract_task.status != TaskStatus.COMPLETED:
                raise Exception(f"Extraction failed: {extract_task.error}")
            
            fields = extract_task.output_data.get("fields", {})
            
            # Log extraction results
            for field_name, field_data in fields.items():
                self.log_reasoning(state, "Extractor Agent",
                    f"Extracted {field_name}",
                    f"Value: {field_data.get('value')}",
                    f"Confidence: {field_data.get('confidence', 0):.0%}, Method: {field_data.get('method', 'unknown')}")
            
            # Phase 4: Validation
            state.current_phase = "validation"
            self.log_reasoning(state, "Coordinator",
                "Fields extracted. Validating results.",
                "Dispatching to Validator agent",
                "Checking value ranges and consistency")
            
            validate_task = await self._execute_validation(document_id, fields, ocr_text)
            state.tasks["validation"] = validate_task
            
            # Phase 5: Self-Correction (if needed)
            if validate_task.status == TaskStatus.NEEDS_REVIEW:
                state.current_phase = "correction"
                self.log_reasoning(state, "Validator Agent",
                    f"Validation issues found: {validate_task.output_data.get('issues', [])}",
                    "Attempting self-correction",
                    "Re-extracting problematic fields...")
                
                fields = await self._self_correct(fields, validate_task.output_data, ocr_text, image)
            
            # Phase 6: Synthesis
            state.current_phase = "synthesis"
            self.log_reasoning(state, "Coordinator",
                "All agents complete. Synthesizing final result.",
                "Combining field extraction and vision results",
                "Calculating overall confidence")
            
            # Combine all results
            final_result = self._synthesize_results(
                fields=fields,
                vision_result=vision_task.output_data,
                validation=validate_task.output_data
            )
            
            # Add agent reasoning chains
            final_result["agentic_reasoning"] = {
                "workflow_log": state.reasoning_log,
                "agents_used": [role.value for role in self.agents.keys()],
                "phases_completed": ["planning", "extraction", "field_extraction", "validation", "synthesis"],
                "total_reasoning_steps": len(state.reasoning_log)
            }
            
            state.final_result = final_result
            state.end_time = datetime.now().isoformat()
            state.current_phase = "completed"
            
            return final_result
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            state.current_phase = "failed"
            self.log_reasoning(state, "Coordinator",
                f"Error occurred: {str(e)}",
                "Handling failure",
                "Returning partial results if available")
            
            return {
                "status": "error",
                "error": str(e),
                "partial_results": state.tasks,
                "reasoning_log": state.reasoning_log
            }
    
    def _create_execution_plan(self, image: np.ndarray) -> Dict:
        """Create execution plan based on document analysis"""
        plan = {
            "parallel_phase": ["ocr", "vision"],
            "sequential_phase": ["extraction", "validation"],
            "optional_phase": ["correction", "llm_enhancement"]
        }
        return plan
    
    def _create_ocr_task(self, doc_id: str, image: np.ndarray) -> AgentTask:
        return AgentTask(
            name="ocr_extraction",
            description="Extract text from document using OCR",
            assigned_to=AgentRole.OCR_SPECIALIST,
            input_data={"image": image}
        )
    
    def _create_vision_task(self, doc_id: str, image: np.ndarray) -> AgentTask:
        return AgentTask(
            name="vision_analysis",
            description="Detect signatures and stamps in document",
            assigned_to=AgentRole.VISION_ANALYST,
            input_data={"image": image}
        )
    
    async def _execute_parallel(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Execute multiple tasks in parallel"""
        async def run_task(task: AgentTask) -> AgentTask:
            agent = self.agents.get(task.assigned_to)
            if agent:
                return await agent.process(task)
            task.status = TaskStatus.FAILED
            task.error = f"No agent found for role {task.assigned_to}"
            return task
        
        results = await asyncio.gather(*[run_task(t) for t in tasks])
        return list(results)
    
    async def _execute_extraction(self, doc_id: str, text: str, image: np.ndarray) -> AgentTask:
        """Execute field extraction"""
        task = AgentTask(
            name="field_extraction",
            description="Extract structured fields from OCR text",
            assigned_to=AgentRole.FIELD_EXTRACTOR,
            input_data={"text": text, "image": image}
        )
        
        agent = self.agents.get(AgentRole.FIELD_EXTRACTOR)
        if agent:
            return await agent.process(task)
        
        task.status = TaskStatus.FAILED
        task.error = "Field extraction agent not available"
        return task
    
    async def _execute_validation(self, doc_id: str, fields: Dict, text: str) -> AgentTask:
        """Execute validation"""
        task = AgentTask(
            name="validation",
            description="Validate extracted fields",
            assigned_to=AgentRole.VALIDATOR,
            input_data={"fields": fields, "text": text}
        )
        
        agent = self.agents.get(AgentRole.VALIDATOR)
        return await agent.process(task)
    
    async def _self_correct(self, fields: Dict, validation: Dict, text: str, image: np.ndarray) -> Dict:
        """Attempt to self-correct extraction errors"""
        issues = validation.get("issues", [])
        corrected_fields = fields.copy()
        
        for issue in issues:
            if "HP" in issue:
                # Re-extract HP with stricter patterns
                import re
                hp_matches = re.findall(r'(\d{2})\s*[Hh][Pp]', text)
                for hp in hp_matches:
                    hp_val = int(hp)
                    if 20 <= hp_val <= 100:
                        corrected_fields["horse_power"] = {
                            "value": hp_val,
                            "confidence": 0.7,
                            "method": "self_correction"
                        }
                        break
            
            if "Cost" in issue:
                # Re-extract cost
                import re
                costs = re.findall(r'[\d,]+(?:\.\d{2})?', text)
                valid_costs = []
                for c in costs:
                    try:
                        val = float(c.replace(",", ""))
                        if 100000 <= val <= 50000000:
                            valid_costs.append(val)
                    except:
                        pass
                if valid_costs:
                    corrected_fields["asset_cost"] = {
                        "value": max(valid_costs),
                        "confidence": 0.6,
                        "method": "self_correction"
                    }
        
        return corrected_fields
    
    def _synthesize_results(self, fields: Dict, vision_result: Dict, validation: Dict) -> Dict:
        """Synthesize final results from all agents"""
        # Add vision results to fields
        fields["dealer_signature"] = {
            "present": vision_result.get("signature", {}).get("present", False),
            "confidence": vision_result.get("signature", {}).get("confidence", 0),
            "bounding_boxes": [d["bbox"] for d in vision_result.get("signature", {}).get("detections", [])]
        }
        
        fields["dealer_stamp"] = {
            "present": vision_result.get("stamp", {}).get("present", False),
            "confidence": vision_result.get("stamp", {}).get("confidence", 0),
            "bounding_boxes": [d["bbox"] for d in vision_result.get("stamp", {}).get("detections", [])]
        }
        
        # Calculate overall confidence
        confidences = []
        for field_data in fields.values():
            if isinstance(field_data, dict) and "confidence" in field_data:
                confidences.append(field_data["confidence"])
        
        overall_confidence = np.mean(confidences) if confidences else 0
        
        return {
            "fields": fields,
            "validation": validation,
            "overall_confidence": overall_confidence,
            "status": "completed"
        }
    
    def get_workflow_state(self, document_id: str) -> Optional[WorkflowState]:
        """Get current workflow state for a document"""
        return self.workflow_states.get(document_id)
    
    def get_reasoning_explanation(self, document_id: str) -> str:
        """Get human-readable explanation of the reasoning process"""
        state = self.workflow_states.get(document_id)
        if not state:
            return "No workflow found for this document"
        
        explanation = ["## Agentic AI Reasoning Process\n"]
        
        for step in state.reasoning_log:
            explanation.append(f"**{step['agent']}** ({step['timestamp'][:19]})")
            explanation.append(f"- Thought: {step['thought']}")
            explanation.append(f"- Action: {step['action']}")
            explanation.append(f"- Result: {step['result']}\n")
        
        return "\n".join(explanation)
