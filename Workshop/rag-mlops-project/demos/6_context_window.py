import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Demo 6: Context Window", layout="wide")

st.title("📏 Demo 6: Context Window Limits")
st.markdown("**Shows how token limits affect RAG systems**")

# Simulated token counts (rough estimate: 1 token ≈ 4 characters)
def estimate_tokens(text):
    return len(text) // 4

SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in large datasets.",
    "Natural Language Processing allows computers to understand human language and generate human-like responses.",
    "Retrieval Augmented Generation combines information retrieval with text generation to produce accurate responses.",
    "Vector databases store embeddings and enable semantic search based on meaning rather than keywords.",
    "Transformers are neural network architectures that use self-attention mechanisms for processing sequences.",
    "Fine-tuning adapts pre-trained models to specific tasks with smaller datasets and domain-specific knowledge.",
    "Prompt engineering designs inputs to guide LLM behavior and outputs for better performance.",
    "Few-shot learning enables models to learn from just a few examples without extensive training data.",
    "Zero-shot learning allows models to perform tasks they weren't explicitly trained on using instructions.",
]

CONTEXT_LIMITS = {
    "GPT-3.5": 4096,
    "GPT-4": 8192,
    "GPT-4-32k": 32768,
    "Claude-2": 100000,
}

with st.sidebar:
    st.header("⚙️ Configuration")
    model = st.selectbox("Model:", list(CONTEXT_LIMITS.keys()))
    num_docs = st.slider("Documents to Include:", 1, 10, 5)
    query = st.text_input("Query:", "Explain machine learning concepts")
    run_btn = st.button("🚀 Check Token Usage", type="primary", use_container_width=True)
    
    st.divider()
    st.info(f"**{model} Limit**: {CONTEXT_LIMITS[model]:,} tokens")

if run_btn:
    context_limit = CONTEXT_LIMITS[model]
    
    # Calculate tokens
    query_tokens = estimate_tokens(query)
    system_tokens = estimate_tokens("You are a helpful AI assistant. Answer based on the context provided.")
    
    context_docs = SAMPLE_DOCS[:num_docs]
    context_text = "\n\n".join(context_docs)
    context_tokens = estimate_tokens(context_text)
    
    response_tokens = 500  # Reserve for response
    
    total_input_tokens = query_tokens + system_tokens + context_tokens
    total_tokens = total_input_tokens + response_tokens
    
    st.header("📊 Token Usage Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Query", f"{query_tokens} tokens")
    col2.metric("Context", f"{context_tokens} tokens")
    col3.metric("Total Input", f"{total_input_tokens} tokens")
    col4.metric("With Response", f"{total_tokens} tokens")
    
    # Status
    if total_tokens <= context_limit:
        st.success(f"✅ Within limit! Using {(total_tokens/context_limit)*100:.1f}% of context window")
    else:
        st.error(f"❌ Exceeds limit by {total_tokens - context_limit} tokens! Need to truncate.")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📝 Token Breakdown", "💡 Solutions"])
    
    with tab1:
        st.subheader("Context Window Usage")
        
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='System Prompt',
            x=[model],
            y=[system_tokens],
            marker_color='#4444ff'
        ))
        
        fig.add_trace(go.Bar(
            name='Query',
            x=[model],
            y=[query_tokens],
            marker_color='#44ff44'
        ))
        
        fig.add_trace(go.Bar(
            name='Context',
            x=[model],
            y=[context_tokens],
            marker_color='#ffaa44'
        ))
        
        fig.add_trace(go.Bar(
            name='Response (Reserved)',
            x=[model],
            y=[response_tokens],
            marker_color='#ff44ff'
        ))
        
        # Add limit line
        fig.add_hline(y=context_limit, line_dash="dash", line_color="red",
                     annotation_text=f"Limit: {context_limit:,} tokens")
        
        fig.update_layout(
            barmode='stack',
            title=f"Token Distribution for {model}",
            yaxis_title="Tokens",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison across models
        st.subheader("Model Comparison")
        
        model_names = list(CONTEXT_LIMITS.keys())
        limits = list(CONTEXT_LIMITS.values())
        usage_pcts = [(total_tokens/limit)*100 for limit in limits]
        
        fig2 = go.Figure(data=[
            go.Bar(name='Used', x=model_names, y=[total_tokens]*len(model_names), marker_color='#44ff44'),
            go.Bar(name='Available', x=model_names, y=[limit-total_tokens for limit in limits], marker_color='#444444')
        ])
        
        fig2.update_layout(
            barmode='stack',
            title="Context Window Capacity Across Models",
            yaxis_title="Tokens",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Detailed Token Breakdown")
        
        breakdown = {
            "Component": ["System Prompt", "Query", "Context", "Response (Reserved)", "Total"],
            "Tokens": [system_tokens, query_tokens, context_tokens, response_tokens, total_tokens],
            "Percentage": [
                f"{(system_tokens/total_tokens)*100:.1f}%",
                f"{(query_tokens/total_tokens)*100:.1f}%",
                f"{(context_tokens/total_tokens)*100:.1f}%",
                f"{(response_tokens/total_tokens)*100:.1f}%",
                "100.0%"
            ]
        }
        
        st.table(breakdown)
        
        st.divider()
        
        st.subheader("Context Documents")
        for i, doc in enumerate(context_docs, 1):
            doc_tokens = estimate_tokens(doc)
            with st.expander(f"Document {i} ({doc_tokens} tokens)"):
                st.write(doc)
        
        if num_docs < len(SAMPLE_DOCS):
            remaining = len(SAMPLE_DOCS) - num_docs
            st.warning(f"⚠️ {remaining} documents not included due to selection")
    
    with tab3:
        st.subheader("💡 Solutions for Token Limit Issues")
        
        if total_tokens > context_limit:
            st.error("### ❌ Current Status: Exceeding Limit")
            
            st.markdown("""
            ### 🔧 Immediate Solutions:
            
            1. **Reduce Retrieved Documents**
               - Current: {} docs
               - Suggested: {} docs
               - Saves: ~{} tokens
            
            2. **Truncate Context**
               - Keep most relevant docs only
               - Use sliding window approach
            
            3. **Summarize Documents**
               - Pre-process long documents
               - Extract key information only
            
            4. **Upgrade Model**
               - Switch to larger context window
               - GPT-4-32k or Claude-2
            """.format(
                num_docs,
                max(1, num_docs - 2),
                estimate_tokens("\n\n".join(SAMPLE_DOCS[num_docs-2:num_docs]))
            ))
        else:
            st.success("### ✅ Current Status: Within Limit")
            
            remaining = context_limit - total_tokens
            additional_docs = remaining // estimate_tokens(SAMPLE_DOCS[0])
            
            st.markdown(f"""
            ### 📈 Optimization Opportunities:
            
            1. **Add More Context**
               - Remaining capacity: {remaining:,} tokens
               - Could add ~{additional_docs} more documents
            
            2. **Increase Response Length**
               - Currently reserved: {response_tokens} tokens
               - Could allow longer responses
            
            3. **Current Efficiency**
               - Using {(total_tokens/context_limit)*100:.1f}% of capacity
               - Good balance for most use cases
            """)
        
        st.divider()
        
        st.markdown("""
        ### 🎯 Best Practices:
        
        1. **Monitor Token Usage**
           - Track input + output tokens
           - Set alerts for approaching limits
        
        2. **Implement Truncation Strategy**
           - Prioritize most relevant content
           - Keep query and response space
        
        3. **Choose Right Model**
           - Balance cost vs context needs
           - Larger context = higher cost
        
        4. **Optimize Context**
           - Remove redundant information
           - Compress where possible
           - Use summarization for long docs
        """)

else:
    st.info("👈 Configure settings and click 'Check Token Usage' to analyze!")
    
    st.markdown("""
    ## 🎯 What This Demo Shows:
    
    ### Context Window Limits:
    - **GPT-3.5**: 4K tokens (~3K words)
    - **GPT-4**: 8K tokens (~6K words)
    - **GPT-4-32k**: 32K tokens (~24K words)
    - **Claude-2**: 100K tokens (~75K words)
    
    ### Token Composition:
    1. **System Prompt**: Instructions to LLM
    2. **Query**: User's question
    3. **Context**: Retrieved documents
    4. **Response**: Generated answer
    
    ### Why This Matters:
    - Exceeding limits = Truncation or errors
    - More context = Better answers (up to a point)
    - Larger windows = Higher costs
    
    ## 💡 Key Takeaway:
    **Balance context size with model limits and costs!**
    """)

st.divider()
st.caption("📏 Context Window Demo - Shows token limit management")
