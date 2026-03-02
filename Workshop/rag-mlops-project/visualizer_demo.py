import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

st.set_page_config(page_title="RAG Demo Visualizer", layout="wide")

st.title("🎨 RAG Pipeline Visualizer - DEMO MODE")
st.markdown("**No API keys needed!** This demo shows how RAG works with sample data.")

# Initialize embedding model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample documents
SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in large datasets.",
    "Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language.",
    "Retrieval Augmented Generation (RAG) combines information retrieval with text generation to produce accurate responses.",
    "Vector databases store embeddings and enable semantic search based on meaning rather than keywords.",
]

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_area("Your Question:", "What is machine learning?", height=100)
    top_k = st.slider("Top K Documents:", 1, 5, 3)
    run_btn = st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True)
    
    st.divider()
    st.info("💡 **Demo Mode**: Using sample AI/ML documents. No API key required!")

if run_btn:
    model = load_model()
    
    # Pipeline Flow
    st.header("🔄 RAG Pipeline Execution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Step 1: Embedding
    with col1:
        st.markdown("### 1️⃣ Query")
        st.info(f"**Input:** {query[:40]}...")
        with st.spinner("Embedding..."):
            start = time.time()
            query_emb = model.encode([query])[0]
            emb_time = time.time() - start
        st.success(f"✅ {emb_time:.3f}s")
        st.caption(f"Vector: {len(query_emb)}D")
    
    # Step 2: Retrieval
    with col2:
        st.markdown("### 2️⃣ Retrieval")
        st.info("**Vector Search**")
        with st.spinner("Searching..."):
            start = time.time()
            doc_embs = model.encode(SAMPLE_DOCS)
            similarities = cosine_similarity([query_emb], doc_embs)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            search_time = time.time() - start
        st.success(f"✅ {search_time:.3f}s")
        st.caption(f"Found: {top_k} docs")
    
    # Step 3: Context
    with col3:
        st.markdown("### 3️⃣ Context")
        st.info("**Prompt Building**")
        context = "\n\n".join([SAMPLE_DOCS[i] for i in top_indices])
        st.success(f"✅ Ready")
        st.caption(f"Chars: {len(context)}")
    
    # Step 4: Generation (simulated)
    with col4:
        st.markdown("### 4️⃣ Generation")
        st.info("**LLM Response**")
        with st.spinner("Generating..."):
            gen_time = 0.8  # Simulated
            time.sleep(0.5)
            # Simple answer based on top doc
            answer = f"Based on the retrieved context: {SAMPLE_DOCS[top_indices[0]]}"
        st.success(f"✅ {gen_time:.2f}s")
        st.caption(f"Tokens: ~{len(answer.split())*1.3:.0f}")
    
    st.divider()
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Similarity Heatmap", 
        "🗺️ Embedding Space", 
        "📝 Retrieved Docs",
        "🎯 Final Output"
    ])
    
    with tab1:
        st.subheader("Cosine Similarity Matrix")
        
        # Calculate full similarity matrix
        all_embs = np.vstack([[query_emb], doc_embs])
        sim_matrix = cosine_similarity(all_embs)
        
        labels = ['Query'] + [f'Doc {i+1}' for i in range(len(SAMPLE_DOCS))]
        
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=labels,
            y=labels,
            colorscale='RdYlGn',
            text=np.round(sim_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Similarity")
        ))
        
        fig.update_layout(
            title="Semantic Similarity Between Query and Documents",
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top matches
        st.subheader("📈 Top Matches")
        for idx in top_indices:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**Doc {idx+1}**")
            with col_b:
                st.metric("Score", f"{similarities[idx]:.4f}")
    
    with tab2:
        st.subheader("3D Embedding Space Visualization")
        
        # PCA to 3D
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(all_embs)
        
        colors = ['red'] + ['blue'] * len(SAMPLE_DOCS)
        sizes = [20] + [12] * len(SAMPLE_DOCS)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers+text',
            marker=dict(size=sizes, color=colors, opacity=0.8),
            text=labels,
            textposition='top center',
            hovertext=[query] + SAMPLE_DOCS,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title="Query (Red) vs Documents (Blue) in 3D Space",
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Closer points = More similar meaning**. The query (red star) is closest to the most relevant documents.")
    
    with tab3:
        st.subheader("📄 Retrieved Context Documents")
        
        for rank, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            
            with st.container():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"### 📄 Rank {rank} - Document {idx+1}")
                with col_b:
                    st.metric("Similarity", f"{similarity:.4f}")
                
                st.text_area(
                    f"Content",
                    SAMPLE_DOCS[idx],
                    height=100,
                    key=f"doc_{idx}",
                    label_visibility="collapsed"
                )
                
                st.divider()
    
    with tab4:
        st.subheader("🤖 Generated Answer")
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <h4>Answer:</h4>
            <p style="font-size: 16px;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ In production, this would be a full LLM-generated response using the retrieved context.")
        
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
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Query Length", f"{len(query.split())} words")
        with col_b:
            st.metric("Context Length", f"{len(context)} chars")
        with col_c:
            st.metric("Avg Similarity", f"{np.mean([similarities[i] for i in top_indices]):.4f}")

else:
    st.info("👈 **Click 'Run RAG Pipeline' in the sidebar to start!**")
    
    st.subheader("📚 What This Demo Shows:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 RAG Process
        1. **Query Embedding**: Convert question to 384D vector
        2. **Vector Search**: Find similar documents using cosine similarity
        3. **Context Building**: Combine top-K documents
        4. **Generation**: LLM creates answer from context
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Visualizations
        - **Similarity Heatmap**: See how close query is to each doc
        - **3D Embedding Space**: Visual representation of semantic similarity
        - **Retrieved Docs**: Actual context used for generation
        - **Metrics**: Performance breakdown
        """)
    
    st.divider()
    
    st.subheader("🚀 Sample Documents in Database:")
    for i, doc in enumerate(SAMPLE_DOCS, 1):
        with st.expander(f"Document {i}"):
            st.write(doc)
    
    st.divider()
    
    st.markdown("""
    ### 🔧 To Use With Real Data:
    1. Get OpenAI API key from https://platform.openai.com/api-keys
    2. Add to `.env` file: `OPENAI_API_KEY=your_key_here`
    3. Run `python visualizer_advanced.py` for full version
    """)

# Footer
st.divider()
st.caption("💡 This is a demo showing RAG concepts. Try different questions to see how semantic search works!")
