# Quick Start Guide

## Prerequisites
1. Install Python 3.11+ from https://www.python.org/downloads/
2. Check "Add Python to PATH" during installation
3. Get OpenAI API key from https://platform.openai.com/api-keys

## Installation

```bash
cd rag-mlops-project
pip install -r requirements.txt
```

## Configuration

```bash
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Run Demo

### Option 1: Python Script
```bash
python -c "
from src.rag_pipeline import RAGPipeline
import mlflow

mlflow.set_tracking_uri('file:./mlruns')
pipeline = RAGPipeline()
pipeline.ingest_documents(['./data/sample.txt'])
result = pipeline.query('What is RAG?')
print(result['answer'])
"
```

### Option 2: API Server
```bash
# Terminal 1: Start MLflow
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: Start API
uvicorn src.api:app --reload

# Terminal 3: Test API
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What is RAG?\"}"
```

### Option 3: Docker
```bash
cd deployment
docker-compose up -d
```

## Test

```bash
pytest tests/ -v
```

## Access Points
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090

## Next Steps
1. Read `docs/CONCEPTS.md` for detailed explanations
2. Explore `notebooks/demo.ipynb` for interactive demo
3. Check MLflow UI for experiment tracking
4. Review Prometheus metrics at `/metrics` endpoint
