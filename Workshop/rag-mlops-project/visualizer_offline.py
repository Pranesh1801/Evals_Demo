import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="RAG Visualizer - Offline", layout="wide")

st.title("🎨 RAG Pipeline Visualizer - OFFLINE MODE")
st.markdown("**Works completely offline!** Using simple embeddings for demo.")

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF based embedding
@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(max_features=384, stop_words='english')

def create_embeddings(texts, vectorizer=None, fit=False):
    """Create TF-IDF embeddings"""
    if vectorizer is None:
        vectorizer = get_vectorizer()
    
    if fit:
        embeddings = vectorizer.fit_transform(texts).toarray()
    else:
        embeddings = vectorizer.transform(texts).toarray()
    
    return embeddings, vectorizer

# Sample documents
SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in large datasets.",
    "Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language.",
    "Retrieval Augmented Generation (RAG) combines information retrieval with text generation to produce accurate responses.",
    "Vector databases store embeddings and enable semantic search based on meaning rather than keywords.",
]

# Pre-compute document embeddings
@st.cache_data
def get_doc_embeddings():
    vectorizer = get_vectorizer()
    embeddings, _ = create_embeddings(SAMPLE_DOCS, vectorizer, fit=True)
    return embeddings, vectorizer

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_area("Your Question:", "What is machine learning?", height=100)
    top_k = st.slider("Top K Documents:", 1, 5, 3)
    run_btn = st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True)
    
    st.divider()
    st.success("✅ **Offline Mode**: No internet needed!")

if run_btn:
    doc_embs, vectorizer = get_doc_embeddings()
    
    # Pipeline Flow
    st.header("🔄 RAG Pipeline Execution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Step 1: Embedding
    with col1:
        st.markdown("### 1️⃣ Query")
        st.info(f"**Input:** {query[:40]}...")
        with st.spinner("Embedding..."):
            start = time.time()
            query_emb, _ = create_embeddings([query], vectorizer, fit=False)
            query_emb = query_emb[0]
            emb_time = time.time() - start
        st.success(f"✅ {emb_time:.3f}s")
        st.caption(f"Vector: {len(query_emb)}D")
    
    # Step 2: Retrieval
    with col2:
        st.markdown("### 2️⃣ Retrieval")
        st.info("**Vector Search**")
        with st.spinner("Searching..."):
            start = time.time()
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
    
    # Step 4: Generation
    with col4:
        st.markdown("### 4️⃣ Generation")
        st.info("**LLM Response**")
        with st.spinner("Generating..."):
            gen_time = 0.8
            time.sleep(0.5)
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
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Top Matches")
        for idx in top_indices:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**Doc {idx+1}**")
            with col_b:
                st.metric("Score", f"{similarities[idx]:.4f}")
    
    with tab2:
        st.subheader("3D Embedding Space Visualization")
        
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
        
        st.info("💡 **Closer points = More similar meaning**")
    
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
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <h4 style="color: #ffffff; margin-top: 0;">Answer:</h4>
            <p style="font-size: 16px; color: #ffffff;">{answer}</p>
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
        1. **Query Embedding**: Convert question to vector
        2. **Vector Search**: Find similar documents
        3. **Context Building**: Combine top-K documents
        4. **Generation**: Create answer from context
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Visualizations
        - **Similarity Heatmap**: Query vs documents
        - **3D Embedding Space**: Visual similarity
        - **Retrieved Docs**: Context used
        - **Metrics**: Performance breakdown
        """)
    
    st.divider()
    
    st.subheader("🚀 Sample Documents:")
    for i, doc in enumerate(SAMPLE_DOCS, 1):
        with st.expander(f"Document {i}"):
            st.write(doc)

st.divider()
st.caption("💡 Offline demo - No internet or API keys needed!")
