# DisasterMind

A multi-agent agentic AI system for disaster prediction and response planning, targeting cyclone and earthquake events across India. Built using LangGraph, FastAPI, and Groq-hosted LLaMA models.

Developed as part of a research internship at NIT Delhi.

---

## Overview

DisasterMind takes a location as input, fetches live meteorological and seismic data, classifies the likely disaster type, predicts severity using trained ML models, retrieves official guidance from IMD and NDMA documents via a RAG pipeline, generates an operational briefing for a duty officer, and produces a structured PDF incident report — all in a single pipeline invocation.

---

## Architecture

The system is implemented as a LangGraph state graph with five specialized agents:

**Data Agent** — Geocodes the input location using the OpenWeatherMap API, fetches live wind speed and pressure data, queries the USGS earthquake feed for recent seismic activity within a 300 km radius, and populates the shared state with all features required by downstream agents.

**Risk Assessment Agent** — Routes the input to a disaster-type classifier (cyclone vs earthquake), then runs the appropriate severity model. Uses SHAP TreeExplainer to extract feature-level contributions for explainability.

**RAG Agent** — Queries a ChromaDB vector store built from four official PDFs (IMD and NDMA guidelines for cyclone and earthquake) using MMR retrieval. Constructs a semantically rich query from the predicted disaster type, severity, and top SHAP factors.

**Planning Agent** — Generates a 4-6 sentence operational briefing using a Groq-hosted LLaMA model, grounded strictly in the retrieved guidance context.

**Report Agent** — Produces a PDF incident report using ReportLab, including a classification summary table, SHAP factor table, operational briefing, and source citations.

```
Input Location
     |
Data Agent  -->  [is_valid?]  -->  reject
                      |
                   Route
                  /      \
     Cyclone Severity   Earthquake Severity
                  \      /
                  RAG Agent
                     |
               Planning Agent
                     |
               Report Agent
                     |
                  PDF Output
```

---

## ML Models

Three models are trained and serialized with joblib:

| Model | Task | Algorithm | Performance |
|-------|------|-----------|-------------|
| model1 | Cyclone vs Earthquake classifier | Random Forest | Macro F1: 0.92 |
| model2 | Cyclone severity prediction | Gradient Boosted Trees | F1: 0.66 |
| model3 | Earthquake severity prediction | Pipeline (OrdinalEncoder + GBT) | F1: 0.64 |

**Datasets:** IBTrACS (cyclone tracks) and USGS earthquake catalog for severity labels.

**Severity labels:**

- Cyclone: Low, Moderate, Severe, Very Severe
- Earthquake: Minor, Moderate, Strong, Major, Great

The cascaded architecture — first classifying disaster type, then predicting severity within that class — is the core architectural novelty of this system.

---

## RAG Pipeline

**Documents ingested:**
- IMD Cyclone Warning SOP
- IMD Earthquake Guidelines
- NDMA Cyclone Management Guidelines
- NDMA Earthquake Guidelines

**Stack:** pdfplumber for extraction, RecursiveCharacterTextSplitter (chunk size 1000, overlap 200), sentence-transformers/all-MiniLM-L6-v2 for embeddings, ChromaDB for vector storage, MMR retrieval with k=8.

**RAGAS Evaluation Results (6 domain-specific test questions):**

| Metric | Score |
|--------|-------|
| Context Recall | 0.667 |
| Faithfulness | 0.797 |
| Factual Correctness | 0.517 |
| Context Precision | 0.696 |

---

## Project Structure

```
DisasterMind/
├── agents/
│   ├── data_agent.py           # Geocoding, OWM weather, USGS seismic fetch
│   ├── risk_assessment_agent.py # Disaster type routing + severity models
│   ├── rag_agent.py            # ChromaDB retrieval
│   ├── planning_agent.py       # LLM briefing generation
│   └── report_agent.py         # PDF report generation
├── rag/
│   ├── ingest.py               # PDF ingestion and embedding pipeline
│   ├── dataset/                # Source PDFs (IMD + NDMA)
│   └── vector_db/              # ChromaDB persistent store
├── models/                     # Serialized joblib model files
├── report/                     # Generated PDF reports
├── orchestrator.py             # LangGraph graph definition and compilation
├── app.py                      # FastAPI REST API
├── config.py                   # API key loading from .env
├── evaluation.py               # RAGAS evaluation pipeline
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.11, a virtual environment, OWM API key, Groq API key.

```bash
git clone https://github.com/Abhi2939/DisasterMind
cd DisasterMind
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OWM_API_KEY=your_openweathermap_key
GROQ_API_KEY=your_groq_key
```

Ingest the RAG documents:

```bash
python rag/ingest.py
```

---

## Running

**Direct pipeline test:**

```bash
python orchestrator.py
```

This runs the full pipeline for Visakhapatnam by default and dumps the complete state along with the report path.

**FastAPI server:**

```bash
uvicorn app:app --reload
```

API is available at `http://127.0.0.1:8000`. Swagger UI at `/docs`.

**POST /predict**

```json
{
  "location": "Visakhapatnam, India"
}
```

**Response:**

```json
{
  "is_valid": true,
  "validation_errors": [],
  "disaster_type": "cyclone",
  "severity": "Low",
  "severity_confidence": 0.987,
  "briefing": "...",
  "report_path": "report/incident_report_cyclone_20260626_112001.pdf",
  "report_url": "/report/incident_report_cyclone_20260626_112001.pdf"
}
```

**RAGAS evaluation:**

```bash
python evaluation.py
```

Results saved to `rag_evaluation_results.csv`.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Run full disaster prediction pipeline |
| `/reports/{filename}` | GET | Download a generated PDF report |

---

## External APIs

| API | Purpose | Notes |
|-----|---------|-------|
| OpenWeatherMap Geocoding + Current Weather | Location resolution, wind speed, pressure | Free tier |
| USGS Earthquake Feed | Recent seismic events within radius | Public, no key required |
| Groq (LLaMA 3.1 8B Instant) | Briefing generation per agent | Free tier, rate limited |

---

## Known Limitations

- Cyclone subbasin detection covers Bay of Bengal (BB) and Arabian Sea (AS) only. Locations outside these basins default to None.
- USGS seismic fallback uses regional historical depth profiles when no live event is detected within 300 km in the past 30 days. Magnitude is not available in fallback mode.
- ML models were trained on scikit-learn 1.6.1. Running on later versions may produce version mismatch warnings.
- RAG source documents are procedural and operational in nature. Severity-specific response guidance is limited in the source PDFs, which constrains factual correctness scores.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent orchestration | LangGraph |
| LLM inference | Groq (LLaMA 3.1 8B Instant) |
| ML models | scikit-learn (Random Forest, Gradient Boosted Trees) |
| Explainability | SHAP TreeExplainer |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| PDF ingestion | pdfplumber |
| Report generation | ReportLab |
| API serving | FastAPI + Uvicorn |
| RAG evaluation | RAGAS |

---

## Author

Kumar Abhinandan
Electronics and Communication Engineering, Thapar Institute of Engineering and Technology (Batch 2024-2028)
Research Intern, Climate Lab, NIT Delhi
GitHub: https://github.com/Abhi2939
