from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
from prometheus_client import Counter, Histogram, generate_latest
import time

app = FastAPI(title="RAG API")
pipeline = RAGPipeline()

query_counter = Counter('rag_queries_total', 'Total queries')
latency_histogram = Histogram('rag_latency_seconds', 'Query latency')

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    query_counter.inc()
    start = time.time()
    result = pipeline.query(request.question)
    latency_histogram.observe(time.time() - start)
    return result

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)
    pipeline.ingest_documents([file_path])
    return {"status": "success", "file": file.filename}

@app.get("/metrics")
async def metrics():
    return generate_latest()

@app.get("/health")
async def health():
    return {"status": "healthy"}
