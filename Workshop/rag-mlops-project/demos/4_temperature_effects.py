import streamlit as st
import time
import random

st.set_page_config(page_title="Demo 4: Temperature Effects", layout="wide")

st.title("🌡️ Demo 4: Temperature Effects")
st.markdown("**Shows how temperature controls creativity vs consistency in LLM responses**")

CONTEXT = "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming."

with st.sidebar:
    st.header("⚙️ Configuration")
    query = st.text_input("Query:", "Explain machine learning")
    run_btn = st.button("🚀 Generate Responses", type="primary", use_container_width=True)
    
    st.divider()
    st.info("**Temperature**: Controls randomness in generation")

if run_btn:
    st.header(f"🔍 Query: {query}")
    st.markdown(f"**Context**: {CONTEXT}")
    
    st.divider()
    
    temperatures = [0.0, 0.5, 1.0, 1.5]
    cols = st.columns(4)
    
    responses = {
        0.0: "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming. It uses algorithms to identify patterns and make decisions.",
        0.5: "Machine learning is a branch of AI that allows systems to learn from data automatically. It's used in applications like recommendation systems, image recognition, and predictive analytics.",
        1.0: "Think of machine learning as teaching computers to learn like humans do - through experience! Instead of programming every rule, we feed data to algorithms that discover patterns on their own. Pretty cool, right?",
        1.5: "Machine learning? It's like magic for computers! Imagine a system that gets smarter just by looking at examples - no hand-holding needed. From Netflix recommendations to self-driving cars, ML is revolutionizing everything. The future is here!"
    }
    
    for col, temp in zip(cols, temperatures):
        with col:
            st.subheader(f"🌡️ Temp = {temp}")
            
            with st.spinner("Generating..."):
                time.sleep(1)
            
            response = responses[temp]
            
            # Color code based on temperature
            if temp == 0.0:
                border_color = "#4444ff"
                label = "Deterministic"
            elif temp == 0.5:
                border_color = "#44ff44"
                label = "Balanced"
            elif temp == 1.0:
                border_color = "#ffaa44"
                label = "Creative"
            else:
                border_color = "#ff4444"
                label = "Very Creative"
            
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {border_color}; min-height: 200px;">
                <p style="color: #ffffff; font-size: 14px;">{response}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"**{label}**")
            
            # Characteristics
            if temp == 0.0:
                st.markdown("✅ Consistent\n\n✅ Factual\n\n❌ Repetitive")
            elif temp == 0.5:
                st.markdown("✅ Reliable\n\n✅ Varied\n\n✅ Balanced")
            elif temp == 1.0:
                st.markdown("✅ Engaging\n\n⚠️ Less predictable\n\n✅ Natural")
            else:
                st.markdown("✅ Very creative\n\n❌ Inconsistent\n\n❌ May hallucinate")
    
    st.divider()
    
    st.subheader("📊 Temperature Comparison")
    
    comparison_data = {
        "Temperature": ["0.0", "0.5", "1.0", "1.5"],
        "Consistency": ["Very High", "High", "Medium", "Low"],
        "Creativity": ["Very Low", "Medium", "High", "Very High"],
        "Factuality": ["Very High", "High", "Medium", "Low"],
        "Use Case": ["Facts/Data", "General Q&A", "Creative Writing", "Brainstorming"]
    }
    
    st.table(comparison_data)
    
    st.divider()
    
    st.subheader("💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ When to Use Low Temperature (0.0-0.3):
        - Factual Q&A
        - Data extraction
        - Code generation
        - Mathematical problems
        - Consistent formatting
        
        **Example**: "What is 2+2?" → Always "4"
        """)
    
    with col2:
        st.markdown("""
        ### ✅ When to Use High Temperature (0.8-1.5):
        - Creative writing
        - Brainstorming ideas
        - Diverse responses
        - Marketing copy
        - Storytelling
        
        **Example**: "Write a tagline" → Many variations
        """)
    
    st.success("""
    ### 🎯 Best Practice for RAG:
    **Use Temperature = 0.7**
    - Balanced between consistency and naturalness
    - Reliable for most Q&A scenarios
    - Reduces hallucinations while staying engaging
    """)

else:
    st.info("👈 Click 'Generate Responses' to see temperature effects!")
    
    st.markdown("""
    ## 🎯 What This Demo Shows:
    
    ### Temperature Scale:
    - **0.0**: Deterministic, always same output
    - **0.5**: Balanced, reliable variation
    - **1.0**: Creative, natural language
    - **1.5+**: Very creative, unpredictable
    
    ### The Trade-off:
    - **Low Temp**: Consistent but boring
    - **High Temp**: Creative but risky
    - **Medium Temp**: Best of both worlds ✅
    
    ### Technical Details:
    Temperature controls the probability distribution:
    - **Low**: Picks most likely tokens (deterministic)
    - **High**: Samples from broader distribution (random)
    
    ## 💡 Key Takeaway:
    **Temperature is a dial between consistency and creativity!**
    """)

st.divider()
st.caption("🌡️ Temperature Effects Demo - Shows LLM parameter tuning")
