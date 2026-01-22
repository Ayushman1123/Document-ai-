# Document AI System - Invoice Field Extraction

🚀 **Intelligent Document AI for Field Extraction from Invoices**

An end-to-end AI-powered system that extracts key fields from invoice documents (tractor loan quotations, retail invoices, etc.) with 95%+ accuracy.

## ✨ Features

- **Multi-language OCR**: Supports English, Hindi, Gujarati, Tamil, Telugu, Marathi
- **Field Extraction**: Dealer Name, Model Name, Horse Power, Asset Cost
- **Visual Detection**: Signature and Stamp detection with bounding boxes
- **Agentic AI**: Self-improving extraction with feedback learning
- **Confidence Scoring**: Every field comes with confidence scores
- **Modern Web UI**: Beautiful dark theme with glassmorphism design

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Web UI)                       │
│  - Document Upload  - Results Display  - Analytics Dashboard │
└────────────────────────────┬────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ OCR Service  │ │Vision Service│ │ Field Extractor      │ │
│  │ (PaddleOCR)  │ │ (YOLO)       │ │ (Pattern+Fuzzy+LLM)  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Agentic AI Layer                      ││
│  │  - Strategy Planning  - Validation  - Self-Correction   ││
│  │  - Feedback Learning  - Explainable Reasoning           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 📋 Extracted Fields

| Field | Type | Evaluation |
|-------|------|------------|
| Dealer Name | Text | Fuzzy match ≥90% |
| Model Name | Text | Exact match |
| Horse Power | Numeric | Exact (±5% tolerance) |
| Asset Cost | Numeric | Exact (±5% tolerance) |
| Dealer Signature | Binary + BBox | IoU ≥0.5 |
| Dealer Stamp | Binary + BBox | IoU ≥0.5 |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (optional, for development server)
- GPU (optional, for faster processing)

### Installation

1. **Clone and setup backend:**

```bash
cd document-ai-system/backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Configure environment:**

```bash
copy .env.example .env
# Edit .env and add your API keys (optional for LLM features)
```

3. **Run the backend:**

```bash
python main.py
```

4. **Open frontend:**

Open `frontend/index.html` in your browser, or use:

```bash
cd frontend
python -m http.server 5500
```

Then visit: http://localhost:5500

## 🔧 API Reference

### Extract Fields
```
POST /api/extract
Content-Type: multipart/form-data

Parameters:
- file: PDF or image file
- use_agentic: boolean (enable AI reasoning)

Response:
{
  "document_id": "abc123",
  "fields": {
    "dealer_name": {"value": "...", "confidence": 0.95},
    "model_name": {"value": "...", "confidence": 0.92},
    ...
  },
  "metadata": {
    "processing_time_seconds": 2.5,
    "overall_confidence": 0.89
  }
}
```

### Submit Feedback
```
POST /api/feedback
{
  "document_id": "abc123",
  "field_name": "dealer_name",
  "predicted_value": "Old Value",
  "correct_value": "Correct Value"
}
```

## 📊 Performance Metrics

- **Target Accuracy**: ≥95% Document-Level Accuracy
- **Processing Time**: <30 seconds per document
- **Cost per Document**: <$0.01

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Uvicorn
- **OCR**: PaddleOCR (multilingual)
- **Vision**: YOLOv8, OpenCV
- **LLM**: OpenAI GPT-4V / Google Gemini (optional)
- **Frontend**: HTML5, CSS3, JavaScript

## 📁 Project Structure

```
document-ai-system/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── requirements.txt     # Python dependencies
│   └── services/
│       ├── ocr_service.py       # OCR extraction
│       ├── vision_service.py    # Signature/stamp detection
│       ├── field_extractor.py   # Field extraction logic
│       ├── document_processor.py # Main pipeline
│       └── agentic_ai.py        # AI reasoning layer
├── frontend/
│   ├── index.html           # Main page
│   ├── styles.css           # Styling
│   └── app.js               # Frontend logic
├── data/
│   ├── uploads/             # Uploaded documents
│   └── output/              # Processing results
└── models/                  # ML model weights
```

## 🤖 Agentic AI Features

The system includes an intelligent agent that:

1. **Plans Extraction Strategy** - Analyzes document characteristics
2. **Multi-Strategy Extraction** - Uses pattern matching, fuzzy matching, and LLM
3. **Self-Validation** - Verifies extracted values
4. **Auto-Correction** - Fixes errors using multiple methods
5. **Feedback Learning** - Improves from user corrections
6. **Explainable Output** - Provides reasoning for extractions

## 📝 License

MIT License - feel free to use for your hackathon!

## 🙏 Acknowledgments

- PaddleOCR for multilingual OCR
- Ultralytics for YOLOv8
- OpenAI/Google for vision LLMs
