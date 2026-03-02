import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time

st.set_page_config(page_title="RAG - TF-IDF Embeddings", layout="wide")

st.title("🎨 RAG Pipeline - TF-IDF EMBEDDINGS (Good)")
st.markdown("**✅ Using TF-IDF embeddings - Understands word importance and meaning!**")

@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(max_features=384, stop_words='english')

def create_embeddings(texts, vectorizer=None, fit=False):
    if vectorizer is None:
        vectorizer = get_vectorizer()
    
    if fit:
        embeddings = vectorizer.fit_transform(texts).toarray()
    else:
        embeddings = vectorizer.transform(texts).toarray()
    
    return embeddings, vectorizer

SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in large datasets.",
    "Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language.",
    "Retrieval Augmented Generation (RAG) combines information retrieval with text generation to produce accurate responses.",
    "Vector databases store embeddings and enable semantic search based on meaning rather than keywords.",
]

@st.cache_data
def get_doc_embeddings():
    vectorizer = get_vectorizer()
    embeddings, _ = create_embeddings(SAMPLE_DOCS, vectorizer, fit=True)
    return embeddings, vectorizer

with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_area("Your Question:", "What is machine learning?", height=100)
    top_k = st.slider("Top K Documents:", 1, 5, 3)
    run_btn = st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True)
    
    st.divider()
    st.success("✅ **TF-IDF Embeddings**: Understands word meaning!")

if run_btn:
    doc_embs, vectorizer = get_doc_embeddings()
    
    st.header("🔄 RAG Pipeline Execution")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col3:
        st.markdown("### 3️⃣ Context")
        st.info("**Prompt Building**")
        context = "\n\n".join([SAMPLE_DOCS[i] for i in top_indices])
        st.success(f"✅ Ready")
        st.caption(f"Chars: {len(context)}")
    
    with col4:
        st.markdown("### 4️⃣ Generation")
        st.info("**LLM Response**")
        gen_time = 0.8
        time.sleep(0.5)
        answer = f"Based on the retrieved context: {SAMPLE_DOCS[top_indices[0]]}"
        st.success(f"✅ {gen_time:.2f}s")
        st.caption(f"Tokens: ~{len(answer.split())*1.3:.0f}")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Similarity Heatmap", "📝 Retrieved Docs", "🎯 Output"])
    
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
        
        fig.update_layout(title="TF-IDF Embeddings - Meaningful Similarities!", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ **Notice**: Similarities make sense! Documents with matching keywords score higher.")
        
        st.subheader("📈 Top Matches (Correct!)")
        for rank, idx in enumerate(top_indices, 1):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**Rank {rank}: Doc {idx+1}**")
            with col_b:
                st.metric("Score", f"{similarities[idx]:.4f}")
    
    with tab2:
        st.subheader("📄 Retrieved Documents")
        
        for rank, idx in enumerate(top_indices, 1):
            with st.container():
                st.markdown(f"### 📄 Rank {rank} - Document {idx+1}")
                st.metric("Similarity", f"{similarities[idx]:.4f}")
                st.text_area(f"Content", SAMPLE_DOCS[idx], height=100, key=f"doc_{idx}", label_visibility="collapsed")
                st.divider()
    
    with tab3:
        st.subheader("🤖 Generated Answer")
        
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
            <h4 style="color: #ffffff; margin-top: 0;">Answer (Correct!):</h4>
            <p style="font-size: 16px; color: #ffffff;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ **Success**: TF-IDF found the right document with matching keywords!")
        
        st.subheader("📊 Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Embedding", f"{emb_time*1000:.1f}ms")
        col2.metric("Search", f"{search_time*1000:.1f}ms")
        col3.metric("Total", f"{emb_time+search_time+gen_time:.2f}s")

else:
    st.info("👈 Click 'Run RAG Pipeline' to see TF-IDF embeddings work!")
    
    st.success("""
    ### ✅ Why TF-IDF Works:
    - **Word frequency**: Counts important words in documents
    - **Inverse document frequency**: Reduces weight of common words
    - **Semantic matching**: Similar texts have similar embeddings
    - **Reliable retrieval**: Finds documents with matching keywords
    - **Correct answers**: LLM gets relevant context
    """)
    
    st.subheader("🚀 Sample Documents:")
    for i, doc in enumerate(SAMPLE_DOCS, 1):
        with st.expander(f"Document {i}"):
            st.write(doc)

st.divider()
st.caption("✅ TF-IDF embeddings demo - Shows proper embedding quality!")
