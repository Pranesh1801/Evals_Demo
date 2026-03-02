import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

st.set_page_config(page_title="PDF Q&A: Vector vs Vectorless RAG", layout="wide")

st.title("📄 PDF Q&A: Vector vs Vectorless RAG")
st.markdown("**Upload a PDF and compare two RAG approaches**")

# BM25 Implementation (Vectorless)
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(documents)
        
        # Build inverted index
        self.inverted_index = {}
        for doc_id, doc in enumerate(documents):
            words = set(doc.lower().split())
            for word in words:
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(doc_id)
    
    def score(self, query, doc_id):
        score = 0
        doc_length = self.doc_lengths[doc_id]
        doc_words = self.documents[doc_id].lower().split()
        
        for term in query.lower().split():
            if term not in self.inverted_index:
                continue
            
            # Term frequency in document
            tf = doc_words.count(term)
            
            # Document frequency
            df = len(self.inverted_index[term])
            
            # IDF
            idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query, top_k=3):
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Smart answer generation with document understanding
def generate_answer(query, context_chunks, all_chunks, method="vector"):
    """Generate intelligent answer with document understanding"""
    combined_context = " ".join(context_chunks)
    query_lower = query.lower()
    
    # Extract key information from RETRIEVED context (not all chunks)
    sentences = [s.strip() for s in combined_context.split('.') if len(s.strip()) > 15]
    
    if any(word in query_lower for word in ['about', 'topic', 'summary', 'main']):
        # Document summary - use FIRST chunks for overview
        first_chunks = all_chunks[:3]
        combined_first = " ".join(first_chunks)
        
        # Extract key phrases
        import re
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', combined_first)
        
        # Get most common meaningful words
        word_freq = {}
        for word in combined_first.lower().split():
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics = [w for w, _ in top_words]
        
        # Build summary
        if capitalized:
            answer = f"This document is about {capitalized[0]}. "
        else:
            answer = "This document discusses "
        
        answer += f"Key topics include: {', '.join(topics)}. "
        
        # Add first meaningful sentence
        if sentences:
            answer += sentences[0] + "."
        
        return answer
    
    # For ALL other queries, use ONLY the retrieved context
    elif any(word in query_lower for word in ['who', 'person', 'name', 'author']):
        # Extract names from RETRIEVED chunks
        import re
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', combined_context)
        if names:
            return f"According to the document: {', '.join(set(names[:3]))}. {sentences[0] if sentences else ''}"
        return f"{sentences[0] if sentences else combined_context[:300]}"
    
    elif any(word in query_lower for word in ['what is', 'define', 'explain', 'tell me about']):
        # Definition - use RETRIEVED context only
        key_terms = [w for w in query_lower.split() if len(w) > 3 and w not in ['what', 'define', 'explain', 'tell', 'about', 'this']]
        
        # Find sentences mentioning the key terms
        relevant_sentences = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in key_terms):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:3]) + "."
        
        # Fallback to first sentences from retrieved context
        return ". ".join(sentences[:3]) + "." if sentences else "No relevant information found in the retrieved sections."
    
    elif any(word in query_lower for word in ['how', 'process', 'steps', 'work']):
        # Process explanation from RETRIEVED context
        relevant_sentences = [s for s in sentences if any(word in s.lower() for word in ['step', 'process', 'first', 'then', 'how', 'using', 'by'])]
        if relevant_sentences:
            return ". ".join(relevant_sentences[:3]) + "."
        return ". ".join(sentences[:3]) + "." if sentences else "No relevant information found."
    
    else:
        # General query - return RETRIEVED context sentences
        if sentences:
            return ". ".join(sentences[:3]) + "."
        return "No relevant information found in the retrieved sections."
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# PDF extraction
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

with st.sidebar:
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        if st.button("🔄 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Extracting text from PDF..."):
                text = extract_pdf_text(uploaded_file)
                st.session_state.chunks = chunk_text(text)
                st.session_state.pdf_processed = True
                st.success(f"✅ Extracted {len(st.session_state.chunks)} chunks!")
    
    st.divider()
    
    if st.session_state.pdf_processed:
        st.header("❓ Ask Questions")
        query = st.text_area("Your Question:", "What is the main topic?", height=100)
        top_k = st.slider("Top-K Chunks:", 1, 5, 3)
        
        compare_btn = st.button("🔍 Compare Both Approaches", type="primary", use_container_width=True)
    else:
        st.info("Upload and process a PDF first!")
        compare_btn = False

if st.session_state.pdf_processed and compare_btn:
    chunks = st.session_state.chunks
    
    st.header(f"🔍 Query: {query}")
    st.markdown(f"**Document**: {len(chunks)} chunks | **Top-K**: {top_k}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    # VECTOR RAG (TF-IDF)
    with col1:
        st.subheader("🎯 Vector RAG (Semantic)")
        
        with st.spinner("Creating embeddings..."):
            start = time.time()
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            doc_embeddings = vectorizer.fit_transform(chunks).toarray()
            query_embedding = vectorizer.transform([query]).toarray()[0]
            emb_time = time.time() - start
        
        with st.spinner("Searching..."):
            start = time.time()
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            top_indices_vector = np.argsort(similarities)[::-1][:top_k]
            search_time = time.time() - start
        
        # Generate intelligent answer
        gen_start = time.time()
        vector_context_chunks = [chunks[i] for i in top_indices_vector]
        vector_answer = generate_answer(query, vector_context_chunks, chunks, "vector")
        gen_time = time.time() - gen_start
        
        total_vector_time = emb_time + search_time + gen_time
        
        # Display metrics
        st.markdown("### ⏱️ Performance")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Embedding", f"{emb_time*1000:.0f}ms")
        metric_col2.metric("Search", f"{search_time*1000:.0f}ms")
        st.metric("Total Time", f"{total_vector_time:.2f}s")
        
        # Display answer
        st.markdown("### 💬 Answer")
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #4444ff;">
            <p style="color: #ffffff; font-size: 14px;">{vector_answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Retrieved chunks
        with st.expander("📝 Retrieved Chunks"):
            for rank, idx in enumerate(top_indices_vector, 1):
                st.markdown(f"**Rank {rank}** (Similarity: {similarities[idx]:.4f})")
                st.text_area(f"Chunk {idx}", chunks[idx][:300], height=100, key=f"vec_{idx}", label_visibility="collapsed")
                st.divider()
    
    # VECTORLESS RAG (BM25)
    with col2:
        st.subheader("🔑 Vectorless RAG (Keyword)")
        
        with st.spinner("Building index..."):
            start = time.time()
            bm25 = BM25(chunks)
            index_time = time.time() - start
        
        with st.spinner("Searching..."):
            start = time.time()
            bm25_results = bm25.search(query, top_k)
            search_time = time.time() - start
        
        # Generate intelligent answer
        gen_start = time.time()
        top_indices_bm25 = [doc_id for doc_id, _ in bm25_results]
        bm25_context_chunks = [chunks[i] for i in top_indices_bm25]
        bm25_answer = generate_answer(query, bm25_context_chunks, chunks, "bm25")
        gen_time = time.time() - gen_start
        
        total_bm25_time = index_time + search_time + gen_time
        
        # Display metrics
        st.markdown("### ⏱️ Performance")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Indexing", f"{index_time*1000:.0f}ms")
        metric_col2.metric("Search", f"{search_time*1000:.0f}ms")
        st.metric("Total Time", f"{total_bm25_time:.2f}s")
        
        # Display answer
        st.markdown("### 💬 Answer")
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #44ff44;">
            <p style="color: #ffffff; font-size: 14px;">{bm25_answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Retrieved chunks
        with st.expander("📝 Retrieved Chunks"):
            for rank, (idx, score) in enumerate(bm25_results, 1):
                st.markdown(f"**Rank {rank}** (BM25 Score: {score:.4f})")
                st.text_area(f"Chunk {idx}", chunks[idx][:300], height=100, key=f"bm25_{idx}", label_visibility="collapsed")
                st.divider()
    
    st.divider()
    
    # Comparison Analysis
    st.header("📊 Comparison Analysis")
    
    tab1, tab2, tab3 = st.tabs(["⚡ Performance", "🎯 Accuracy", "💡 Recommendations"])
    
    with tab1:
        st.subheader("Performance Comparison")
        
        fig = go.Figure(data=[
            go.Bar(name='Vector RAG', x=['Embedding/Index', 'Search', 'Total'], 
                   y=[emb_time*1000, search_time*1000, total_vector_time*1000],
                   marker_color='#4444ff'),
            go.Bar(name='Vectorless RAG', x=['Embedding/Index', 'Search', 'Total'], 
                   y=[index_time*1000, search_time*1000, total_bm25_time*1000],
                   marker_color='#44ff44')
        ])
        
        fig.update_layout(
            title="Latency Comparison (milliseconds)",
            barmode='group',
            height=400,
            yaxis_title="Time (ms)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 🎯 Vector RAG")
            st.markdown(f"""
            - **Embedding**: {emb_time*1000:.0f}ms
            - **Search**: {search_time*1000:.0f}ms
            - **Total**: {total_vector_time:.2f}s
            - **Overhead**: Embedding computation
            """)
        
        with col_b:
            st.markdown("### 🔑 Vectorless RAG")
            st.markdown(f"""
            - **Indexing**: {index_time*1000:.0f}ms
            - **Search**: {search_time*1000:.0f}ms
            - **Total**: {total_bm25_time:.2f}s
            - **Overhead**: Index building
            """)
    
    with tab2:
        st.subheader("Retrieval Accuracy")
        
        # Check overlap
        overlap = set(top_indices_vector) & set(top_indices_bm25)
        overlap_pct = len(overlap) / top_k * 100
        
        st.metric("Chunk Overlap", f"{len(overlap)}/{top_k} ({overlap_pct:.0f}%)")
        
        if overlap_pct > 66:
            st.success("✅ High agreement between methods - Both found similar content!")
        elif overlap_pct > 33:
            st.info("⚠️ Moderate agreement - Methods found different but potentially relevant content")
        else:
            st.warning("❌ Low agreement - Methods retrieved very different content")
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 🎯 Vector RAG Retrieved")
            for idx in top_indices_vector:
                if idx in overlap:
                    st.success(f"✅ Chunk {idx} (Both methods)")
                else:
                    st.info(f"📄 Chunk {idx} (Vector only)")
        
        with col_b:
            st.markdown("### 🔑 Vectorless RAG Retrieved")
            for idx in top_indices_bm25:
                if idx in overlap:
                    st.success(f"✅ Chunk {idx} (Both methods)")
                else:
                    st.info(f"📄 Chunk {idx} (BM25 only)")
    
    with tab3:
        st.subheader("💡 When to Use Each Approach")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            ### 🎯 Use Vector RAG When:
            
            ✅ **Semantic understanding needed**
            - "What causes rain?" (conceptual)
            - Synonyms matter ("car" = "automobile")
            - Paraphrased queries
            
            ✅ **General Q&A**
            - Customer support
            - Knowledge bases
            - Educational content
            
            ✅ **Multilingual**
            - Cross-language search
            - Translation needed
            
            ⚠️ **Trade-offs:**
            - Slower (embedding overhead)
            - More complex infrastructure
            - Higher costs
            """)
        
        with col_b:
            st.markdown("""
            ### 🔑 Use Vectorless RAG When:
            
            ✅ **Exact terms matter**
            - Technical documentation
            - Legal/medical docs
            - API references
            
            ✅ **Speed critical**
            - Real-time search
            - High throughput needed
            - Low latency required
            
            ✅ **Simple infrastructure**
            - Limited resources
            - Easy deployment
            - Lower costs
            
            ⚠️ **Trade-offs:**
            - Misses synonyms
            - No semantic understanding
            - Keyword-dependent
            """)
        
        st.divider()
        
        st.success("""
        ### 🚀 Best Practice: Hybrid Approach
        
        Combine both methods for optimal results:
        1. **BM25** for fast exact matches
        2. **Vector** for semantic understanding
        3. **Merge & re-rank** results
        4. **Get best of both worlds!**
        """)

elif st.session_state.pdf_processed:
    st.info("👈 Enter your question in the sidebar and click 'Compare Both Approaches'!")
    
    st.markdown(f"""
    ## 📄 PDF Processed Successfully!
    
    **Chunks extracted**: {len(st.session_state.chunks)}
    
    ### 🎯 What This Demo Shows:
    
    1. **Vector RAG (TF-IDF)**
       - Semantic search using embeddings
       - Understands meaning and context
       - Better for conceptual queries
    
    2. **Vectorless RAG (BM25)**
       - Keyword-based search
       - Fast and simple
       - Better for exact term matching
    
    3. **Side-by-Side Comparison**
       - Performance metrics
       - Retrieval accuracy
       - When to use each
    
    ### 💡 Try Different Queries:
    - Conceptual: "What is the main idea?"
    - Specific: "What is the definition of X?"
    - Technical: "How does Y work?"
    """)

else:
    st.info("👈 Upload a PDF file to get started!")
    
    st.markdown("""
    ## 🎯 How This Demo Works:
    
    ### Step 1: Upload PDF
    - Upload any PDF document
    - Text is extracted automatically
    - Document is split into chunks
    
    ### Step 2: Ask Questions
    - Enter your question
    - Select number of chunks to retrieve
    - Click to compare both approaches
    
    ### Step 3: Compare Results
    - See Vector RAG (semantic search)
    - See Vectorless RAG (keyword search)
    - Analyze performance and accuracy
    
    ## 📊 What You'll Learn:
    
    ### Vector RAG:
    - Uses TF-IDF embeddings
    - Semantic similarity search
    - Better for meaning-based queries
    
    ### Vectorless RAG:
    - Uses BM25 algorithm
    - Keyword matching
    - Better for exact term queries
    
    ### Comparison:
    - Performance metrics
    - Retrieval overlap
    - When to use each approach
    
    ## 💡 Try It With:
    - Research papers
    - Technical documentation
    - Business reports
    - Any PDF document!
    """)

st.divider()
st.caption("📄 PDF Q&A Demo - Compare Vector vs Vectorless RAG approaches")
