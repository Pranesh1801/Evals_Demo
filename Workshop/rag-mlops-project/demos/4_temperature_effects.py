import streamlit as st
import numpy as np
import time

st.set_page_config(page_title="Demo 4: Temperature Effects", layout="wide")

st.title("🌡️ Demo 4: Temperature Effects")
st.markdown("**Real probabilistic sampling - see how temperature changes token selection**")

# Phrase bank with logits (preferences)
PHRASE_BANK = {
    "opening": [
        ("Machine learning is", 5.0),
        ("ML is", 2.0),
        ("Think of machine learning as", 1.5),
        ("Imagine ML as", 0.8),
        ("Machine learning? It's like", 0.5)
    ],
    "definition": [
        ("a subset of AI that enables systems to learn from data", 5.0),
        ("teaching computers to learn from examples", 3.0),
        ("giving computers the ability to learn without explicit programming", 2.5),
        ("magic that lets computers get smarter over time", 1.0),
        ("the art of making machines intelligent through data", 0.8)
    ],
    "mechanism": [
        ("It uses algorithms to identify patterns.", 5.0),
        ("Algorithms discover patterns automatically.", 3.5),
        ("The system learns by finding patterns in data.", 2.5),
        ("It's all about pattern recognition, baby!", 1.0),
        ("Neural networks do their magic behind the scenes.", 0.7)
    ],
    "application": [
        ("Used in recommendation systems and image recognition.", 5.0),
        ("Powers everything from Netflix to self-driving cars.", 3.0),
        ("You see it everywhere - Siri, Alexa, your phone's camera!", 2.0),
        ("It's revolutionizing literally everything around us.", 1.2),
        ("From predicting weather to diagnosing diseases - ML does it all!", 0.9)
    ],
    "closer": [
        ("It's a fundamental technology in modern AI.", 5.0),
        ("Pretty powerful stuff.", 3.0),
        ("The future is here!", 1.5),
        ("Mind-blowing, right?", 1.0),
        ("Welcome to the AI revolution! 🚀", 0.6)
    ]
}

def softmax(logits, temperature):
    """Apply temperature and compute softmax probabilities"""
    if temperature == 0:
        # Deterministic: pick highest logit
        probs = np.zeros(len(logits))
        probs[np.argmax(logits)] = 1.0
        return probs
    scaled = np.array(logits) / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))  # numerical stability
    return exp_scaled / exp_scaled.sum()

def generate_response(temperature, seed=42):
    """Generate response by sampling from phrase bank"""
    np.random.seed(seed)
    parts = []
    
    for role, phrases in PHRASE_BANK.items():
        texts, logits = zip(*phrases)
        probs = softmax(logits, temperature)
        chosen_idx = np.random.choice(len(texts), p=probs)
        parts.append(texts[chosen_idx])
    
    return " ".join(parts)

with st.sidebar:
    st.header("⚙️ Configuration")
    seed = st.number_input("Random Seed:", 0, 100, 42, help="Change to see different samples")
    run_btn = st.button("🎲 Sample Responses", type="primary", use_container_width=True)
    
    st.divider()
    st.info("**Temperature**: Controls sampling randomness via softmax")

if run_btn:
    st.header("🎲 Probabilistic Sampling Demo")
    
    st.divider()
    
    temperatures = [0.0, 0.5, 1.0, 1.5]
    cols = st.columns(4)
    
    for col, temp in zip(cols, temperatures):
        with col:
            st.subheader(f"🌡️ Temp = {temp}")
            
            with st.spinner("Sampling..."):
                time.sleep(0.5)
            
            response = generate_response(temp, seed)
            
            border_color = ["#4444ff", "#44ff44", "#ffaa44", "#ff4444"][temperatures.index(temp)]
            label = ["Deterministic", "Balanced", "Creative", "Very Creative"][temperatures.index(temp)]
            
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

    # Show probability distributions
    st.divider()
    st.subheader("📊 Probability Distributions")
    
    import plotly.graph_objects as go
    
    role = "opening"
    texts, logits = zip(*PHRASE_BANK[role])
    
    fig = go.Figure()
    for temp in temperatures:
        probs = softmax(logits, temp if temp > 0 else 0.01)
        fig.add_trace(go.Bar(
            name=f"T={temp}",
            x=list(range(len(texts))),
            y=probs,
            text=[f"{p:.2f}" for p in probs],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f"Sampling Probabilities for '{role}' phrases",
        xaxis_title="Phrase Index",
        yaxis_title="Probability",
        barmode='group',
        height=400,
        xaxis=dict(tickmode='array', tickvals=list(range(len(texts))), ticktext=[f"P{i+1}" for i in range(len(texts))])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📝 View Phrase Options"):
        for i, (text, logit) in enumerate(zip(texts, logits), 1):
            st.markdown(f"**P{i}** (logit={logit}): {text}")

else:
    st.info("👈 Click 'Sample Responses' to see real temperature sampling!")
    
    st.markdown("""
    ## 🎯 How This Works:
    
    ### Real Probabilistic Sampling:
    1. **Phrase Bank**: 5 roles × 5 phrases each with base logits
    2. **Temperature Scaling**: `logits / T`
    3. **Softmax**: Convert to probabilities
    4. **Sample**: Pick phrases based on probabilities
    5. **Concatenate**: Build complete response
    
    ### Temperature Effects:
    - **T=0.0**: Always picks highest logit (deterministic)
    - **T=0.5**: Strongly favors high logits, some variety
    - **T=1.0**: Balanced sampling from distribution
    - **T=1.5**: Flatter distribution, more randomness
    
    ### Try It:
    - Click 'Sample Responses' multiple times
    - Change the seed to see different samples
    - Notice how T=0.0 never changes
    - Notice how T=1.5 varies wildly
    
    ## 💡 Key Insight:
    **This is exactly how LLMs work - they sample next tokens from probability distributions controlled by temperature!**
    """)

st.divider()
st.caption("🌡️ Temperature Effects Demo - Real probabilistic sampling with softmax")
