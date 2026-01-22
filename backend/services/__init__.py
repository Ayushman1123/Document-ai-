"""
Document AI Services Package
"""
from .ocr_service import OCRService
from .vision_service import VisionService
from .field_extractor import FieldExtractor
from .document_processor import DocumentProcessor, DocumentMetrics
from .agentic_ai import AgenticAI
from .orchestration import (
    AgentRole, TaskStatus, AgentTask, AgentMessage,
    BaseAgent, OCRAgent, VisionAgent, ExtractorAgent, ValidatorAgent
)
from .coordinator import AgenticCoordinator
from .auth_service import AuthService, get_auth_service

__all__ = [
    "OCRService",
    "VisionService", 
    "FieldExtractor",
    "DocumentProcessor",
    "DocumentMetrics",
    "AgenticAI",
    "AgenticCoordinator",
    "AgentRole",
    "TaskStatus",
    "AgentTask",
    "AuthService",
    "get_auth_service"
]

