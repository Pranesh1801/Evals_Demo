import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.rag_pipeline import RAGPipeline
import time
import json

st.set_page_config(page_title="Advanced RAG Visualizer", layout="wide")

st.title("🎨 Advanced RAG Pipeline Visualizer")
st.markdown("Real-time visualization of Retrieval Augmented Generation")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_area("Query:", "What is machine learning?", height=100)
    top_k = st.slider("Top K:", 1, 5, 3)
    show_embeddings = st.checkbox("Show Raw Embeddings", False)
    run_btn = st.button("🚀 Execute Pipeline", type="primary", use_container_width=True)

if run_btn:
    # Pipeline Flow Diagram
    st.header("🔄 Pipeline Flow")
    
    flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
    
    with flow_col1:
        st.markdown("### 1️⃣ Query")
        st.info(f"**Input:** {query[:50]}...")
        with st.spinner("Embedding..."):
            start = time.time()
            query_emb = pipeline.embedding_model.embed_query(query)
            emb_time = time.time() - start
        st.success(f"✅ {emb_time:.3f}s")
        st.caption(f"Vector: {len(query_emb)}D")
    
    with flow_col2:
        st.markdown("### 2️⃣ Retrieval")
        st.info("**Vector Search**")
        with st.spinner("Searching..."):
            start = time.time()
            results = pipeline.vector_store.search(query, top_k)
            search_time = time.time() - start
        st.success(f"✅ {search_time:.3f}s")
        st.caption(f"Found: {len(results)} docs")
    
    with flow_col3:
        st.markdown("### 3️⃣ Context")
        st.info("**Prompt Building**")
        context_text = "\n".join([r['text'][:100] for r in results])
        st.success(f"✅ Ready")
        st.caption(f"Chars: {len(context_text)}")
    
    with flow_col4:
        st.markdown("### 4️⃣ Generation")
        st.info("**LLM Response**")
        with st.spinner("Generating..."):
            start = time.time()
            answer = pipeline.llm.generate_response(query, results)
            gen_time = time.time() - start
        st.success(f"✅ {gen_time:.3f}s")
        st.caption(f"Tokens: ~{len(answer.split())*1.3:.0f}")
    
    st.divider()
    
    # Main Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Similarity Heatmap", 
        "🗺️ Embedding Space", 
        "📝 Retrieved Docs",
        "🎯 Final Output"
    ])
    
    with tab1:
        st.subheader("Cosine Similarity Matrix")
        
        # Calculate similarities
        doc_texts = [r['text'] for r in results]
        doc_embs = pipeline.embedding_model.embed_documents(doc_texts)
        
        all_embs = np.array([query_emb] + doc_embs)
        sim_matrix = cosine_similarity(all_embs)
        
        labels = ['Query'] + [f'Doc {i+1}' for i in range(len(results))]
        
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=labels,
            y=labels,
            colorscale='RdYlGn',
            text=np.round(sim_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Similarity")
        ))
        
        fig.update_layout(
            title="Semantic Similarity Between Query and Documents",
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Similarity scores
        st.subheader("📈 Relevance Scores")
        for i, score in enumerate(sim_matrix[0, 1:], 1):
            st.metric(f"Document {i}", f"{score:.4f}", 
                     delta=f"{(score-0.5)*100:.1f}% vs baseline")
    
    with tab2:
        st.subheader("3D Embedding Visualization")
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(all_embs)
        
        colors = ['red'] + ['blue'] * len(results)
        sizes = [20] + [12] * len(results)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers+text',
            marker=dict(size=sizes, color=colors),
            text=labels,
            textposition='top center'
        )])
        
        fig.update_layout(
            title="3D PCA Projection of Embeddings",
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if show_embeddings:
            st.subheader("Raw Embeddings")
            st.json({
                "query": query_emb[:10],
                "dimensions": len(query_emb)
            })
    
    with tab3:
        st.subheader("Retrieved Context Documents")
        
        for i, doc in enumerate(results, 1):
            similarity = sim_matrix[0, i]
            
            with st.container():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"### 📄 Document {i}")
                with col_b:
                    st.metric("Similarity", f"{similarity:.4f}")
                
                st.text_area(
                    f"Content {i}",
                    doc['text'],
                    height=150,
                    key=f"doc_{i}",
                    label_visibility="collapsed"
                )
                
                if doc.get('metadata'):
                    with st.expander("Metadata"):
                        st.json(doc['metadata'])
                
                st.divider()
    
    with tab4:
        st.subheader("🤖 Generated Answer")
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <h4>Answer:</h4>
            <p style="font-size: 16px;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("📊 Performance Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Embedding Time", f"{emb_time*1000:.1f}ms")
        with metric_col2:
            st.metric("Search Time", f"{search_time*1000:.1f}ms")
        with metric_col3:
            st.metric("Generation Time", f"{gen_time:.2f}s")
        with metric_col4:
            total_time = emb_time + search_time + gen_time
            st.metric("Total Latency", f"{total_time:.2f}s")
        
        st.subheader("🔍 Pipeline Details")
        
        details = {
            "query": query,
            "query_embedding_dim": len(query_emb),
            "documents_retrieved": len(results),
            "context_length": sum(len(r['text']) for r in results),
            "answer_length": len(answer),
            "avg_similarity": float(np.mean(sim_matrix[0, 1:])),
            "max_similarity": float(np.max(sim_matrix[0, 1:])),
            "timings": {
                "embedding_ms": round(emb_time * 1000, 2),
                "search_ms": round(search_time * 1000, 2),
                "generation_s": round(gen_time, 2),
                "total_s": round(total_time, 2)
            }
        }
        
        st.json(details)

else:
    st.info("👈 Configure settings in sidebar and click 'Execute Pipeline' to visualize RAG process")
    
    # Show example
    st.subheader("What You'll See:")
    st.markdown("""
    - **Pipeline Flow**: Step-by-step execution with timing
    - **Similarity Heatmap**: How close query is to each document
    - **3D Embedding Space**: Visual representation of semantic similarity
    - **Retrieved Documents**: Actual context fed to LLM
    - **Final Answer**: Generated response with metrics
    """)
