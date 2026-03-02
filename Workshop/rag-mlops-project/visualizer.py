import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from src.rag_pipeline import RAGPipeline
import time

st.set_page_config(page_title="RAG Visualizer", layout="wide")
st.title("🔍 RAG Pipeline Visualizer")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Input")
    query = st.text_input("Enter your question:", "What is RAG?")
    top_k = st.slider("Top K results:", 1, 5, 3)
    
    if st.button("🚀 Run RAG Pipeline", type="primary"):
        st.session_state.run = True

if st.session_state.get('run'):
    with col2:
        st.header("🎯 Pipeline Execution")
        
        # Step 1: Query Embedding
        with st.status("Step 1: Embedding Query...", expanded=True) as status:
            start = time.time()
            query_emb = pipeline.embedding_model.embed_query(query)
            st.metric("Embedding Dimension", len(query_emb))
            st.metric("Time", f"{time.time()-start:.3f}s")
            status.update(label="✅ Query Embedded", state="complete")
        
        # Step 2: Vector Search
        with st.status("Step 2: Searching Vector DB...", expanded=True) as status:
            start = time.time()
            results = pipeline.vector_store.search(query, top_k)
            st.metric("Documents Retrieved", len(results))
            st.metric("Time", f"{time.time()-start:.3f}s")
            status.update(label="✅ Documents Retrieved", state="complete")
        
        # Step 3: LLM Generation
        with st.status("Step 3: Generating Response...", expanded=True) as status:
            start = time.time()
            answer = pipeline.llm.generate_response(query, results)
            st.metric("Time", f"{time.time()-start:.3f}s")
            status.update(label="✅ Response Generated", state="complete")
    
    # Visualization Section
    st.header("📊 Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Embedding Space", "Retrieved Context", "Final Answer"])
    
    with tab1:
        st.subheader("2D Projection of Embeddings")
        
        # Get all embeddings
        all_docs = [r['text'] for r in results]
        doc_embs = pipeline.embedding_model.embed_documents(all_docs)
        
        # PCA to 2D
        all_embs = np.array([query_emb] + doc_embs)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(all_embs)
        
        # Plot
        fig = go.Figure()
        
        # Query point
        fig.add_trace(go.Scatter(
            x=[coords_2d[0, 0]], y=[coords_2d[0, 1]],
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='star'),
            text=['Query'],
            textposition='top center',
            name='Query'
        ))
        
        # Document points
        fig.add_trace(go.Scatter(
            x=coords_2d[1:, 0], y=coords_2d[1:, 1],
            mode='markers+text',
            marker=dict(size=15, color='blue'),
            text=[f'Doc {i+1}' for i in range(len(doc_embs))],
            textposition='top center',
            name='Documents'
        ))
        
        fig.update_layout(
            title="Query vs Retrieved Documents in Embedding Space",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Retrieved Context")
        for i, doc in enumerate(results, 1):
            with st.expander(f"📄 Document {i}", expanded=True):
                st.write(doc['text'])
                if doc.get('metadata'):
                    st.json(doc['metadata'])
    
    with tab3:
        st.subheader("🤖 Generated Answer")
        st.success(answer)
        
        st.subheader("📈 Pipeline Metrics")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Query Length", len(query.split()))
        col_b.metric("Context Docs", len(results))
        col_c.metric("Answer Length", len(answer.split()))
