import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="Demo 3: Top-K Selection", layout="wide")

st.title("🎯 Demo 3: Top-K Selection Optimization")
st.markdown("**Shows the trade-off between retrieval quality and latency**")

SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural Language Processing allows computers to understand human language.",
    "Retrieval Augmented Generation combines information retrieval with text generation.",
    "Vector databases store embeddings and enable semantic search.",
    "Transformers are neural network architectures that use self-attention mechanisms.",
    "Fine-tuning adapts pre-trained models to specific tasks with smaller datasets.",
    "Prompt engineering designs inputs to guide LLM behavior and outputs.",
]

@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(max_features=384, stop_words='english')

@st.cache_data
def get_embeddings():
    vectorizer = get_vectorizer()
    embeddings = vectorizer.fit_transform(SAMPLE_DOCS).toarray()
    return embeddings, vectorizer

with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_area("Query:", "What is machine learning?", height=100)
    top_k = st.slider("Top-K Documents:", 1, 8, 3, help="Number of documents to retrieve")
    run_btn = st.button("🚀 Run Retrieval", type="primary", use_container_width=True)
    
    st.divider()
    st.info(f"**Current K**: {top_k} documents")

if run_btn:
    doc_embs, vectorizer = get_embeddings()
    
    st.header("🔄 Retrieval Process")
    
    # Embedding
    start = time.time()
    query_emb = vectorizer.transform([query]).toarray()[0]
    emb_time = time.time() - start
    
    # Search
    start = time.time()
    similarities = cosine_similarity([query_emb], doc_embs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    search_time = time.time() - start
    
    # Simulated generation time (increases with more context)
    gen_time = 0.5 + (top_k * 0.15)
    time.sleep(min(gen_time, 2))
    
    total_time = emb_time + search_time + gen_time
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents Retrieved", top_k)
    col2.metric("Search Time", f"{search_time*1000:.1f}ms")
    col3.metric("Generation Time", f"{gen_time:.2f}s")
    col4.metric("Total Latency", f"{total_time:.2f}s")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Trade-off Analysis", "📝 Retrieved Docs", "💡 Recommendations"])
    
    with tab1:
        st.subheader("Quality vs Latency Trade-off")
        
        # Simulate metrics for different K values
        k_values = list(range(1, 9))
        latencies = [0.5 + (k * 0.15) for k in k_values]
        qualities = [min(0.5 + (k * 0.08), 0.95) for k in k_values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=k_values, y=latencies,
            mode='lines+markers',
            name='Latency (s)',
            line=dict(color='#ff4444', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=k_values, y=qualities,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#44ff44', width=3),
            marker=dict(size=10),
            yaxis='y2'
        ))
        
        # Highlight current K
        fig.add_vline(x=top_k, line_dash="dash", line_color="cyan", 
                     annotation_text=f"Current K={top_k}")
        
        fig.update_layout(
            title="Impact of Top-K on Performance",
            xaxis_title="Top-K Value",
            yaxis_title="Latency (seconds)",
            yaxis2=dict(title="Quality Score", overlaying='y', side='right', range=[0, 1]),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Avg Similarity", f"{np.mean([similarities[i] for i in top_indices]):.4f}")
        with col_b:
            st.metric("Context Length", f"{sum(len(SAMPLE_DOCS[i]) for i in top_indices)} chars")
        with col_c:
            quality_score = min(0.5 + (top_k * 0.08), 0.95)
            st.metric("Quality Score", f"{quality_score:.2f}")
    
    with tab2:
        st.subheader("Retrieved Documents")
        
        for rank, idx in enumerate(top_indices, 1):
            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**Rank {rank}: Document {idx+1}**")
                with col_b:
                    st.metric("Similarity", f"{similarities[idx]:.4f}")
                
                st.text_area(
                    "Content",
                    SAMPLE_DOCS[idx],
                    height=80,
                    key=f"doc_{idx}",
                    label_visibility="collapsed"
                )
                st.divider()
    
    with tab3:
        st.subheader("💡 Recommendations")
        
        if top_k == 1:
            st.warning("""
            **K=1 (Too Low)**
            - ⚠️ May miss relevant information
            - ⚠️ Single point of failure
            - ✅ Fastest response time
            - **Recommendation**: Increase to 3-5 for better coverage
            """)
        elif top_k <= 3:
            st.success("""
            **K=2-3 (Optimal)**
            - ✅ Good balance of quality and speed
            - ✅ Captures main relevant information
            - ✅ Reasonable latency
            - **Recommendation**: Good choice for most use cases!
            """)
        elif top_k <= 5:
            st.info("""
            **K=4-5 (High Quality)**
            - ✅ Comprehensive context
            - ⚠️ Slightly higher latency
            - ⚠️ May include less relevant docs
            - **Recommendation**: Use when accuracy is critical
            """)
        else:
            st.error("""
            **K=6+ (Too High)**
            - ⚠️ Significant latency increase
            - ⚠️ Dilutes context with irrelevant info
            - ⚠️ Higher token costs
            - **Recommendation**: Reduce to 3-5 for better efficiency
            """)
        
        st.divider()
        
        st.markdown("""
        ### 🎯 Best Practices:
        - **Start with K=3** as baseline
        - **Monitor quality metrics** to tune
        - **Consider latency requirements** of your application
        - **Test with real queries** from your domain
        """)

else:
    st.info("👈 Adjust Top-K slider and click 'Run Retrieval' to see the impact!")
    
    st.markdown("""
    ## 🎯 What This Demo Shows:
    
    ### The Trade-off:
    - **Low K (1-2)**: Fast but may miss information
    - **Medium K (3-5)**: Balanced quality and speed ✅
    - **High K (6+)**: Comprehensive but slow
    
    ### Key Factors:
    1. **Latency**: More docs = longer processing
    2. **Quality**: More docs = better coverage (up to a point)
    3. **Cost**: More docs = more tokens = higher cost
    4. **Relevance**: Too many docs can dilute context
    
    ## 💡 Key Takeaway:
    **K=3 is often optimal, but tune based on your specific needs!**
    """)

st.divider()
st.caption("🎯 Top-K Selection Demo - Shows retrieval optimization trade-offs")
