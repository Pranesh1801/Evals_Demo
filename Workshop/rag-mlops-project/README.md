# RAG MLOps Project - Complete GenAI Pipeline

## 🎯 Project Overview
Production-ready RAG system demonstrating MLOps best practices for GenAI applications.

## 🔑 Key GenAI Concepts Covered

### 1. **RAG (Retrieval Augmented Generation)**
- Combines retrieval from knowledge base with LLM generation
- Reduces hallucinations by grounding responses in documents

### 2. **Embeddings**
- Converts text to dense vectors (384-dim with MiniLM)
- Enables semantic similarity search

### 3. **Vector Database (ChromaDB)**
- Stores document embeddings
- Fast similarity search using cosine distance

### 4. **Chunking Strategy**
- Splits documents into 1000-char chunks with 200-char overlap
- Preserves context across boundaries

### 5. **Prompt Engineering**
- Context injection into LLM prompts
- Temperature control for response variability

### 6. **MLOps Pipeline**
- Experiment tracking (MLflow)
- Model versioning
- Performance monitoring (Prometheus)
- CI/CD automation

## 🚀 Setup

```bash
# Install Python 3.11+
# Then install dependencies:
pip install -r requirements.txt

# Configure environment:
cp .env.example .env
# Add your OPENAI_API_KEY

# Run MLflow:
mlflow server --host 0.0.0.0 --port 5000

# Run API:
uvicorn src.api:app --reload

# Run tests:
pytest tests/ --cov=src
```

## 📊 Architecture

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Response
                ↓              ↓              ↓              ↓
            MLflow      Prometheus      MLflow         MLflow
```

## 🐳 Deployment

```bash
cd deployment
docker-compose up -d
```

Access:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090

## 📈 Monitoring Metrics

- Query latency
- Token usage
- Retrieval accuracy
- System throughput

## 🧪 Testing

```bash
pytest tests/ -v
flake8 src/
black src/
```

## 🎓 Learning Path

1. **Embeddings** → `src/embeddings.py`
2. **Vector Store** → `src/vector_store.py`
3. **RAG Pipeline** → `src/rag_pipeline.py`
4. **API Layer** → `src/api.py`
5. **MLOps** → MLflow integration throughout
6. **Deployment** → `deployment/`

## 🎨 Visualizers

### Web Visualizer (Streamlit)
```bash
streamlit run visualizer_advanced.py
```
Features:
- Real-time pipeline execution
- 3D embedding space visualization
- Similarity heatmaps
- Performance metrics

### CLI Visualizer (Terminal)
```bash
python visualizer_cli.py "What is machine learning?"
```
Features:
- Terminal-based visualization
- Similarity scores table
- Step-by-step progress

## 📝 API Usage

```python
import requests

# Ingest document
files = {'file': open('document.pdf', 'rb')}
requests.post('http://localhost:8000/ingest', files=files)

# Query
response = requests.post('http://localhost:8000/query', 
    json={'question': 'What is RAG?'})
print(response.json()['answer'])
```

## 🔧 Key Files

- `src/rag_pipeline.py` - Main orchestration
- `src/vector_store.py` - ChromaDB integration
- `src/llm.py` - OpenAI wrapper
- `src/api.py` - FastAPI endpoints
- `deployment/Dockerfile` - Container config
- `.github/workflows/ci-cd.yml` - CI/CD pipeline
