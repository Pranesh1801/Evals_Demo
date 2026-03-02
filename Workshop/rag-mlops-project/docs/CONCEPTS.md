# GenAI & MLOps Concepts - Deep Dive

## 1. Retrieval Augmented Generation (RAG)

**Problem**: LLMs have knowledge cutoff dates and can hallucinate.

**Solution**: RAG retrieves relevant documents and injects them as context.

**Flow**:
```
Query → Embed → Search Vector DB → Retrieve Top-K → Inject in Prompt → LLM → Answer
```

**Benefits**:
- Up-to-date information
- Domain-specific knowledge
- Reduced hallucinations
- Traceable sources

## 2. Embeddings

**What**: Dense vector representations of text (e.g., 384 dimensions).

**How**: Neural networks trained to place similar text close in vector space.

**Models**:
- `all-MiniLM-L6-v2`: Fast, 384-dim
- `text-embedding-ada-002`: OpenAI, 1536-dim

**Code**:
```python
model.encode("Hello") → [0.23, -0.45, 0.12, ...]
```
## 3. Vector Databases

**Purpose**: Store and search embeddings efficiently.

**ChromaDB Features**:
- Persistent storage
- Cosine similarity search
- Metadata filtering
- HNSW indexing

**Search**:
```python
query_vec = embed("What is AI?")
results = db.search(query_vec, top_k=3)
```

## 4. Chunking Strategies

**Why**: LLMs have token limits; documents are too large.

**Strategies**:
- Fixed size: 1000 chars
- Overlap: 200 chars (preserves context)
- Semantic: Split by paragraphs/sentences

**Trade-offs**:
- Small chunks: Precise but lose context
- Large chunks: Context-rich but less precise

## 5. Prompt Engineering

**Context Injection**:
```
Context: [Retrieved documents]
Question: [User query]
Answer: [LLM generates]
```

**Parameters**:
- Temperature: 0.7 (creativity vs consistency)
- Max tokens: Response length
- System prompt: Behavior instructions

## 6. MLOps for GenAI

### Experiment Tracking (MLflow)
- Log parameters (model, chunk_size)
- Log metrics (latency, tokens)
- Version models
- Compare runs

### Monitoring (Prometheus)
- Query rate
- Latency percentiles
- Error rates
- Token usage

### CI/CD
- Automated testing
- Code quality checks
- Docker builds
- Deployment automation

## 7. Evaluation Metrics

**Retrieval**:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)

**Generation**:
- BLEU, ROUGE (reference-based)
- Perplexity
- Human evaluation

**End-to-End**:
- Latency
- Throughput
- Cost per query

## 8. Production Considerations

**Scalability**:
- Vector DB sharding
- LLM caching
- Load balancing

**Cost Optimization**:
- Smaller embedding models
- Prompt compression
- Batch processing

**Security**:
- API key management
- Input sanitization
- Rate limiting

## 9. Advanced Techniques

**Hybrid Search**: Combine vector + keyword search

**Re-ranking**: Use cross-encoder to re-score results

**Query Expansion**: Generate multiple query variations

**Agentic RAG**: Multi-step reasoning with tools

## 10. Deployment Pipeline

```
Code → Test → Build → Push → Deploy → Monitor
  ↓      ↓      ↓      ↓       ↓        ↓
GitHub  Pytest Docker  ECR   ECS/K8s  Prometheus
```

**Stages**:
1. Lint & format (black, flake8)
2. Unit tests (pytest)
3. Integration tests
4. Build Docker image
5. Push to registry
6. Deploy to staging
7. Smoke tests
8. Deploy to production
9. Monitor metrics
