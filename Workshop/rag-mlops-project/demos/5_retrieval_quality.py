import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Demo 5: Retrieval Quality", layout="wide")

st.title("📈 Demo 5: Retrieval Quality Metrics")
st.markdown("**Shows how to measure and evaluate RAG retrieval performance**")

SAMPLE_DOCS = [
    {"text": "Machine learning is a subset of AI that enables systems to learn from data.", "relevant": True},
    {"text": "Deep learning uses neural networks with multiple layers.", "relevant": True},
    {"text": "Natural Language Processing allows computers to understand human language.", "relevant": False},
    {"text": "Retrieval Augmented Generation combines retrieval with text generation.", "relevant": False},
    {"text": "Supervised learning uses labeled data to train models.", "relevant": True},
    {"text": "Unsupervised learning finds patterns in unlabeled data.", "relevant": True},
    {"text": "Reinforcement learning learns through trial and error.", "relevant": True},
    {"text": "Vector databases store embeddings for semantic search.", "relevant": False},
]

@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(max_features=384, stop_words='english')

@st.cache_data
def get_embeddings():
    vectorizer = get_vectorizer()
    texts = [doc["text"] for doc in SAMPLE_DOCS]
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings, vectorizer

with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_input("Query:", "What is machine learning?")
    top_k = st.slider("Top-K:", 1, 8, 3)
    run_btn = st.button("🚀 Run Retrieval", type="primary", use_container_width=True)
    
    st.divider()
    st.info("**Ground Truth**: 5 docs marked as relevant")

if run_btn:
    doc_embs, vectorizer = get_embeddings()
    
    # Retrieve
    query_emb = vectorizer.transform([query]).toarray()[0]
    similarities = cosine_similarity([query_emb], doc_embs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Calculate metrics
    retrieved_relevant = sum(1 for idx in top_indices if SAMPLE_DOCS[idx]["relevant"])
    total_relevant = sum(1 for doc in SAMPLE_DOCS if doc["relevant"])
    
    precision = retrieved_relevant / top_k if top_k > 0 else 0
    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    st.header("📊 Retrieval Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.2%}", help="% of retrieved docs that are relevant")
    col2.metric("Recall", f"{recall:.2%}", help="% of relevant docs that were retrieved")
    col3.metric("F1 Score", f"{f1:.2%}", help="Harmonic mean of precision and recall")
    col4.metric("Retrieved", f"{retrieved_relevant}/{top_k}")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📝 Retrieved Docs", "💡 Analysis"])
    
    with tab1:
        st.subheader("Retrieval Confusion Matrix")
        
        # Calculate confusion matrix
        true_positives = retrieved_relevant
        false_positives = top_k - retrieved_relevant
        false_negatives = total_relevant - retrieved_relevant
        true_negatives = len(SAMPLE_DOCS) - top_k - false_negatives
        
        fig = go.Figure(data=go.Heatmap(
            z=[[true_positives, false_positives],
               [false_negatives, true_negatives]],
            x=['Predicted Relevant', 'Predicted Not Relevant'],
            y=['Actually Relevant', 'Actually Not Relevant'],
            text=[[f'TP: {true_positives}', f'FP: {false_positives}'],
                  [f'FN: {false_negatives}', f'TN: {true_negatives}']],
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='RdYlGn',
            showscale=False
        ))
        
        fig.update_layout(
            title="Retrieval Performance",
            height=400,
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            **True Positives (TP)**: {true_positives}
            - Retrieved AND relevant ✅
            
            **False Positives (FP)**: {false_positives}
            - Retrieved but NOT relevant ❌
            """)
        
        with col_b:
            st.markdown(f"""
            **False Negatives (FN)**: {false_negatives}
            - NOT retrieved but relevant ❌
            
            **True Negatives (TN)**: {true_negatives}
            - NOT retrieved and NOT relevant ✅
            """)
    
    with tab2:
        st.subheader("Retrieved Documents")
        
        for rank, idx in enumerate(top_indices, 1):
            is_relevant = SAMPLE_DOCS[idx]["relevant"]
            
            with st.container():
                col_a, col_b, col_c = st.columns([3, 1, 1])
                
                with col_a:
                    st.markdown(f"**Rank {rank}: Document {idx+1}**")
                with col_b:
                    st.metric("Similarity", f"{similarities[idx]:.4f}")
                with col_c:
                    if is_relevant:
                        st.success("✅ Relevant")
                    else:
                        st.error("❌ Not Relevant")
                
                st.text_area(
                    "Content",
                    SAMPLE_DOCS[idx]["text"],
                    height=60,
                    key=f"doc_{idx}",
                    label_visibility="collapsed"
                )
                st.divider()
    
    with tab3:
        st.subheader("💡 Performance Analysis")
        
        if precision >= 0.8 and recall >= 0.6:
            st.success("""
            ### ✅ Excellent Retrieval!
            - High precision: Most retrieved docs are relevant
            - Good recall: Found most relevant docs
            - System is working well
            """)
        elif precision >= 0.6:
            st.info("""
            ### ⚠️ Good Precision, Low Recall
            - Retrieved docs are mostly relevant
            - But missing some relevant docs
            - **Solution**: Increase Top-K or improve embeddings
            """)
        elif recall >= 0.6:
            st.warning("""
            ### ⚠️ Good Recall, Low Precision
            - Finding relevant docs
            - But also retrieving irrelevant ones
            - **Solution**: Improve ranking or use re-ranker
            """)
        else:
            st.error("""
            ### ❌ Poor Retrieval
            - Low precision AND recall
            - System needs improvement
            - **Solution**: Better embeddings, query expansion, or hybrid search
            """)
        
        st.divider()
        
        st.markdown("""
        ### 📊 Metric Definitions:
        
        **Precision** = TP / (TP + FP)
        - "Of what I retrieved, how much was relevant?"
        - High precision = Few false positives
        
        **Recall** = TP / (TP + FN)
        - "Of all relevant docs, how many did I find?"
        - High recall = Few false negatives
        
        **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
        - Harmonic mean balancing both metrics
        - Good overall performance indicator
        
        ### 🎯 Target Metrics:
        - **Precision**: > 80% (minimize noise)
        - **Recall**: > 60% (find most relevant)
        - **F1**: > 70% (balanced performance)
        """)

else:
    st.info("👈 Click 'Run Retrieval' to see quality metrics!")
    
    st.markdown("""
    ## 🎯 What This Demo Shows:
    
    ### Evaluation Metrics:
    1. **Precision**: Accuracy of retrieved results
    2. **Recall**: Coverage of relevant documents
    3. **F1 Score**: Overall performance balance
    
    ### Why This Matters:
    - **High Precision**: Users get relevant results
    - **High Recall**: Don't miss important information
    - **Balance**: Trade-off between the two
    
    ### Real-World Impact:
    - Low precision → Users lose trust
    - Low recall → Missing critical info
    - Need both for production RAG!
    
    ## 💡 Key Takeaway:
    **Measure retrieval quality to ensure RAG system reliability!**
    """)

st.divider()
st.caption("📈 Retrieval Quality Demo - Shows evaluation metrics for RAG")
