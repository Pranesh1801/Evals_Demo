import streamlit as st
import time

st.set_page_config(page_title="Demo 2: Hallucination", layout="wide")

st.title("🎭 Demo 2: Hallucination Prevention")
st.markdown("**Shows how RAG prevents LLMs from making up facts**")

SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural Language Processing allows computers to understand human language.",
]

with st.sidebar:
    st.header("⚙️ Try These Queries")
    
    st.markdown("### ✅ In Knowledge Base:")
    if st.button("What is machine learning?", use_container_width=True):
        st.session_state.query = "What is machine learning?"
        st.session_state.run = True
    
    st.markdown("### ❌ NOT in Knowledge Base:")
    if st.button("What is quantum computing?", use_container_width=True):
        st.session_state.query = "What is quantum computing?"
        st.session_state.run = True
    
    if st.button("Who invented blockchain?", use_container_width=True):
        st.session_state.query = "Who invented blockchain?"
        st.session_state.run = True

if st.session_state.get('run'):
    query = st.session_state.query
    
    st.header(f"🔍 Query: {query}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❌ Without RAG (Hallucinates)")
        
        with st.spinner("Generating..."):
            time.sleep(1)
            
            # Simulated hallucination
            if "quantum" in query.lower():
                hallucinated = "Quantum computing uses quantum bits (qubits) that can exist in superposition. It was invented by Richard Feynman in 1982 and uses quantum entanglement to perform calculations exponentially faster than classical computers."
            elif "blockchain" in query.lower():
                hallucinated = "Blockchain was invented by Satoshi Nakamoto in 2008. It's a distributed ledger technology that uses cryptographic hashing to create immutable records of transactions across a peer-to-peer network."
            else:
                hallucinated = "Machine learning was invented by Arthur Samuel in 1959. It uses algorithms like neural networks, decision trees, and support vector machines to learn patterns from data and make predictions."
        
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4444;">
            <p style="color: #ffffff; font-size: 16px;">{hallucinated}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.error("⚠️ **Problem**: LLM invents facts not in knowledge base!")
        
        st.markdown("**Issues:**")
        st.markdown("- Makes up specific dates and names")
        st.markdown("- Sounds confident but may be wrong")
        st.markdown("- No way to verify sources")
    
    with col2:
        st.subheader("✅ With RAG (Grounded)")
        
        with st.spinner("Searching knowledge base..."):
            time.sleep(0.5)
            
            # Check if query is in knowledge base
            query_lower = query.lower()
            relevant = False
            context = ""
            
            for doc in SAMPLE_DOCS:
                if any(word in doc.lower() for word in query_lower.split()):
                    relevant = True
                    context = doc
                    break
        
        with st.spinner("Generating..."):
            time.sleep(1)
            
            if relevant:
                grounded = f"Based on the knowledge base: {context}"
            else:
                grounded = "I don't have information about this topic in my knowledge base. I cannot provide an answer without reliable sources."
        
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #44ff44;">
            <p style="color: #ffffff; font-size: 16px;">{grounded}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if relevant:
            st.success("✅ **Success**: Answer grounded in actual documents!")
        else:
            st.success("✅ **Honest**: Admits when information is not available!")
        
        st.markdown("**Benefits:**")
        st.markdown("- Only uses verified information")
        st.markdown("- Admits knowledge gaps")
        st.markdown("- Traceable to source documents")
    
    st.divider()
    
    st.subheader("📚 Knowledge Base Contents")
    for i, doc in enumerate(SAMPLE_DOCS, 1):
        with st.expander(f"Document {i}"):
            st.write(doc)

else:
    st.info("👈 Click a query button in the sidebar to see the comparison!")
    
    st.markdown("""
    ## 🎯 What This Demo Shows:
    
    ### Without RAG:
    - LLM generates plausible-sounding but potentially false information
    - No source verification
    - Confident even when wrong
    
    ### With RAG:
    - Only uses information from knowledge base
    - Admits when it doesn't know
    - Provides traceable sources
    
    ## 💡 Key Takeaway:
    **RAG prevents hallucinations by grounding responses in verified documents!**
    """)

st.divider()
st.caption("🎭 Hallucination Prevention Demo - Shows why RAG is critical for trustworthy AI")
