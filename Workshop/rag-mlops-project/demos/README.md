# 🎓 RAG MLOps Demo Suite

Complete demonstration suite showing GenAI and MLOps concepts with interactive visualizations.

## 📁 Demo Structure

### Demo 1: Embeddings Comparison
**Files**: `1_embeddings_hash.py`, `1_embeddings_tfidf.py`

**Concept**: Why embedding quality matters

**Shows**:
- Hash embeddings (random, poor)
- TF-IDF embeddings (meaningful, good)
- Side-by-side comparison

**Run**:
```bash
python -m streamlit run demos/1_embeddings_hash.py
python -m streamlit run demos/1_embeddings_tfidf.py
```

**Key Takeaway**: Proper embeddings are critical for semantic search!

---

### Demo 2: Hallucination Prevention
**File**: `2_hallucination_demo.py`

**Concept**: How RAG prevents LLM hallucinations

**Shows**:
- Without RAG: Makes up facts
- With RAG: Grounded in documents
- Honesty when information unavailable

**Run**:
```bash
python -m streamlit run demos/2_hallucination_demo.py
```

**Key Takeaway**: RAG ensures trustworthy, verifiable answers!

---

### Demo 3: Top-K Selection
**File**: `3_topk_selection.py`

**Concept**: Retrieval optimization trade-offs

**Shows**:
- Quality vs latency trade-off
- Interactive K selection (1-8)
- Performance metrics

**Run**:
```bash
python -m streamlit run demos/3_topk_selection.py
```

**Key Takeaway**: K=3 is often optimal for balanced performance!

---

### Demo 4: Temperature Effects
**File**: `4_temperature_effects.py`

**Concept**: LLM creativity vs consistency

**Shows**:
- Temperature 0.0: Deterministic
- Temperature 0.5: Balanced
- Temperature 1.0: Creative
- Temperature 1.5: Very creative

**Run**:
```bash
python -m streamlit run demos/4_temperature_effects.py
```

**Key Takeaway**: Temperature=0.7 balances reliability and naturalness!

---

### Demo 5: Retrieval Quality
**File**: `5_retrieval_quality.py`

**Concept**: Measuring RAG performance

**Shows**:
- Precision, Recall, F1 metrics
- Confusion matrix
- Performance analysis

**Run**:
```bash
python -m streamlit run demos/5_retrieval_quality.py
```

**Key Takeaway**: Measure quality to ensure system reliability!

---

### Demo 6: Context Window Limits
**File**: `6_context_window.py`

**Concept**: Token limits in LLMs

**Shows**:
- Token counting
- Model comparisons (GPT-3.5, GPT-4, Claude)
- Truncation strategies

**Run**:
```bash
python -m streamlit run demos/6_context_window.py
```

**Key Takeaway**: Balance context size with model limits and costs!

---

### Demo 7: PDF Q&A - Vector vs Vectorless RAG 🆕
**File**: `7_pdf_qa_comparison.py`

**Concept**: Compare semantic vs keyword-based RAG

**Shows**:
- Upload any PDF document
- Vector RAG (TF-IDF embeddings)
- Vectorless RAG (BM25 keyword search)
- Side-by-side performance comparison
- When to use each approach

**Run**:
```bash
python -m streamlit run demos/7_pdf_qa_comparison.py
```

**Key Takeaway**: Different RAG approaches excel in different scenarios!

---

### FINAL: Production RAG System
**File**: `FINAL_rag_complete.py`

**Concept**: Complete production-ready RAG

**Shows**:
- All best practices combined
- Full metrics dashboard
- Quality indicators
- Performance tracking

**Run**:
```bash
python -m streamlit run demos/FINAL_rag_complete.py
```

**Key Takeaway**: This is how you build production RAG systems!

---

## 🎯 Recommended Demo Flow

### For Technical Audience:
1. **Embeddings** → Show the foundation
2. **Top-K** → Show optimization
3. **Retrieval Quality** → Show evaluation
4. **FINAL** → Show complete system

### For Business Audience:
1. **Hallucination** → Show the problem RAG solves
2. **Temperature** → Show control over outputs
3. **Context Window** → Show scalability
4. **FINAL** → Show production readiness

### For Students/Learning:
1. **Embeddings** → Understand basics
2. **Hallucination** → Understand RAG value
3. **Top-K** → Understand trade-offs
4. **Temperature** → Understand LLM behavior
5. **Retrieval Quality** → Understand metrics
6. **Context Window** → Understand limits
7. **FINAL** → See it all together

---

## 🚀 Quick Start

### Prerequisites:
```bash
pip install streamlit plotly scikit-learn numpy
```

### Run Any Demo:
```bash
cd rag-mlops-project
python -m streamlit run demos/<demo_file>.py
```

### Run All Demos (Different Ports):
```bash
python -m streamlit run demos/1_embeddings_tfidf.py --server.port 8501 &
python -m streamlit run demos/2_hallucination_demo.py --server.port 8502 &
python -m streamlit run demos/3_topk_selection.py --server.port 8503 &
python -m streamlit run demos/FINAL_rag_complete.py --server.port 8504
```

---

## 📊 Concepts Covered

### GenAI Concepts:
- ✅ Embeddings (TF-IDF vs Hash)
- ✅ Semantic Search
- ✅ RAG Architecture
- ✅ Hallucination Prevention
- ✅ Temperature Control
- ✅ Context Windows
- ✅ Token Management

### MLOps Concepts:
- ✅ Performance Metrics
- ✅ Quality Evaluation (Precision/Recall)
- ✅ Latency Tracking
- ✅ Trade-off Analysis
- ✅ System Monitoring
- ✅ Production Best Practices

---

## 🎓 Learning Objectives

After completing all demos, you will understand:

1. **Why embeddings matter** for semantic search
2. **How RAG prevents** hallucinations
3. **How to optimize** retrieval with Top-K
4. **How to control** LLM behavior with temperature
5. **How to measure** system quality
6. **How to handle** token limits
7. **How to build** production RAG systems

---

## 💡 Tips for Presenting

### Demo 1 (Embeddings):
- Run hash version first to show failure
- Then run TF-IDF to show success
- Emphasize the dramatic difference

### Demo 2 (Hallucination):
- Try queries NOT in knowledge base
- Show how RAG admits "I don't know"
- Contrast with hallucinated responses

### Demo 3 (Top-K):
- Start with K=1, show it's too low
- Increase to K=3, show it's optimal
- Increase to K=8, show diminishing returns

### Demo 4 (Temperature):
- Show all 4 temperatures side-by-side
- Highlight how responses change
- Explain use cases for each

### Demo 5 (Quality):
- Show confusion matrix
- Explain precision vs recall
- Discuss real-world impact

### Demo 6 (Context):
- Show different models
- Demonstrate token overflow
- Discuss cost implications

### FINAL:
- Run complete pipeline
- Show all metrics
- Emphasize production readiness

---

## 🔧 Customization

### Add Your Own Documents:
Edit `SAMPLE_DOCS` in any demo file:
```python
SAMPLE_DOCS = [
    "Your custom document 1",
    "Your custom document 2",
    # ...
]
```

### Change Model Settings:
Adjust in sidebar or code:
```python
top_k = 3  # Number of documents
temperature = 0.7  # LLM creativity
max_tokens = 500  # Response length
```

---

## 📚 Additional Resources

- **Main README**: `../README.md`
- **Concepts Guide**: `../docs/CONCEPTS.md`
- **Quick Start**: `../QUICKSTART.md`

---

## 🎯 Next Steps

1. **Run all demos** to understand concepts
2. **Customize with your data** for your domain
3. **Integrate with real LLM** (OpenAI, etc.)
4. **Add MLflow tracking** for experiments
5. **Deploy to production** with Docker

---

## 💬 Questions?

Each demo is self-contained and includes:
- Interactive controls
- Visual explanations
- Key takeaways
- Best practices

**Experiment, learn, and build amazing RAG systems!** 🚀
