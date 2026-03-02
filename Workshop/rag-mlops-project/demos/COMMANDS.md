# 🚀 Quick Demo Commands

## Run Individual Demos

```bash
# Demo 1: Embeddings Comparison
python -m streamlit run demos/1_embeddings_hash.py
python -m streamlit run demos/1_embeddings_tfidf.py

# Demo 2: Hallucination Prevention
python -m streamlit run demos/2_hallucination_demo.py

# Demo 3: Top-K Selection
python -m streamlit run demos/3_topk_selection.py

# Demo 4: Temperature Effects
python -m streamlit run demos/4_temperature_effects.py

# Demo 5: Retrieval Quality
python -m streamlit run demos/5_retrieval_quality.py

# Demo 6: Context Window
python -m streamlit run demos/6_context_window.py

# FINAL: Production RAG
python -m streamlit run demos/FINAL_rag_complete.py
```

## Run Multiple Demos Simultaneously

```bash
# Run on different ports
python -m streamlit run demos/1_embeddings_tfidf.py --server.port 8501 &
python -m streamlit run demos/2_hallucination_demo.py --server.port 8502 &
python -m streamlit run demos/3_topk_selection.py --server.port 8503 &
python -m streamlit run demos/FINAL_rag_complete.py --server.port 8504
```

## Access URLs

- Demo 1: http://localhost:8501
- Demo 2: http://localhost:8502
- Demo 3: http://localhost:8503
- FINAL: http://localhost:8504

## 🎯 Recommended Demo Order

### For Presentations:
1. Embeddings (show foundation)
2. Hallucination (show value)
3. Top-K (show optimization)
4. FINAL (show complete system)

### For Learning:
1. All demos in order (1-6)
2. FINAL (see it all together)

## 💡 Tips

- Each demo is standalone
- No API keys needed
- All work offline
- Interactive controls in sidebar
- Black background, white text everywhere!
