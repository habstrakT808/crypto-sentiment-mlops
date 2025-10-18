# File: src/monitoring/dashboard/streamlit_app.py
"""
🎨 MODERN MINIMALIST CRYPTO SENTIMENT DASHBOARD
Simple, Beautiful, Professional
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Crypto Sentiment Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIMPLE & MODERN CSS
# ============================================================================

st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background - Dark Theme */
    .main {
        background: #0f0f23;
        color: #ffffff;
    }
    
    /* Sidebar - Dark Blue */
    [data-testid="stSidebar"] {
        background: #1a1d35;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Main Title - High Contrast White */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        margin: 2rem 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Clean Card Design */
    .clean-card {
        background: #1a1d35;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Metric Cards - Simple & Clean */
    .metric-card {
        background: #1a1d35;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    
    .metric-card h3 {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34, 197, 94, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .status-badge-offline {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    
    /* Result Card */
    .result-card {
        background: #1a1d35;
        border-radius: 20px;
        padding: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
        text-align: center;
    }
    
    .result-emoji {
        font-size: 6rem;
        margin-bottom: 1rem;
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1rem 0;
    }
    
    .positive { color: #22c55e; }
    .neutral { color: #f59e0b; }
    .negative { color: #ef4444; }
    
    /* Confidence Bar */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50px;
        height: 50px;
        margin: 2rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-bar {
        height: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 50px;
        transition: width 1s ease-out;
    }
    
    /* Probability Cards */
    .prob-card {
        background: #1a1d35;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .prob-card:hover {
        transform: translateY(-5px);
    }
    
    .prob-card-emoji {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .prob-card-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .prob-card-label {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Input Styling */
    .stTextArea textarea {
        background: #1a1d35 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div {
        background: transparent !important;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div {
        background: #1a1d35 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    /* Info/Warning/Error Boxes */
    .stAlert {
        background: #1a1d35 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d35;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8b5cf6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CLIENT
# ============================================================================

class DashboardAPI:
    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = os.getenv("API_BASE_URL")
            if not base_url:
                base_url = "http://api:8000" if os.path.exists('/.dockerenv') else "http://localhost:8000"
        
        self.base_url = base_url
        self.headers = {"X-API-Key": "demo-key-789", "Content-Type": "application/json"}
    
    def get_health(self):
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", headers=self.headers, timeout=5)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except:
            return {"status": "error"}
    
    def predict_sentiment(self, text: str):
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict-dev",
                json={"text": text, "include_features": True},
                timeout=10
            )
            return response.json() if response.status_code == 200 else {"error": "Failed"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self):
        try:
            response = requests.get(f"{self.base_url}/api/v1/model/info", headers=self.headers, timeout=5)
            return response.json() if response.status_code == 200 else {"error": "Unavailable"}
        except:
            return {"error": "Unavailable"}

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_gauge_chart(confidence: float, prediction: str):
    """Simple gauge chart"""
    colors = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
    color = colors.get(prediction.lower(), "#6366f1")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [66, 100], 'color': 'rgba(34, 197, 94, 0.1)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_bar_chart(probs: dict):
    """Simple bar chart - FIXED"""
    sentiments = ['Positive', 'Neutral', 'Negative']
    values = [probs.get('positive', 0) * 100, probs.get('neutral', 0) * 100, probs.get('negative', 0) * 100]
    colors = ['#22c55e', '#f59e0b', '#ef4444']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sentiments,
        y=values,
        marker=dict(color=colors),
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=16, color='white', family='Inter'),  # ✅ FIXED: Removed 'weight'
        hovertemplate='<b>%{x}</b><br>%{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 29, 53, 0.5)',
        font={'color': 'white', 'family': 'Inter'},
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', showline=False, range=[0, max(values) * 1.2]),
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_feature_chart(features: dict):
    """Feature importance chart - FIXED"""
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    names = [f[0].replace('_', ' ').title() for f in sorted_features]
    values = [f[1] for f in sorted_features]
    colors = ['#22c55e' if v > 0 else '#ef4444' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names[::-1],
        x=values[::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f'{v:.2f}' for v in values[::-1]],
        textposition='outside',
        textfont=dict(size=12, color='white', family='Inter'),  # ✅ FIXED: Removed 'weight'
        hovertemplate='<b>%{y}</b><br>%{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 29, 53, 0.5)',
        font={'color': 'white', 'family': 'Inter'},
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', showline=False),
        yaxis=dict(showgrid=False, showline=False),
        height=400,
        margin=dict(l=150, r=50, t=40, b=40)
    )
    
    return fig

# ============================================================================
# RESULT DISPLAY
# ============================================================================

def show_results(result: dict):
    """Display results - Simple & Clean"""
    prediction = result.get("prediction", "unknown").lower()
    confidence = result.get("confidence", 0)
    probs = result.get("probabilities", {})
    
    emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😞"}
    emoji = emoji_map.get(prediction, "🤔")
    
    # Main result
    st.markdown(f"""
    <div class="result-card">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label {prediction}">{prediction}</div>
        <div class="confidence-bar-container">
            <div class="confidence-bar" style="width: {confidence*100}%">
                {confidence*100:.1f}% Confident
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Probabilities
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-card-emoji">😊</div>
            <div class="prob-card-value" style="color: #22c55e;">{probs.get('positive', 0)*100:.1f}%</div>
            <div class="prob-card-label">Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-card-emoji">😐</div>
            <div class="prob-card-value" style="color: #f59e0b;">{probs.get('neutral', 0)*100:.1f}%</div>
            <div class="prob-card-label">Neutral</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-card-emoji">😞</div>
            <div class="prob-card-value" style="color: #ef4444;">{probs.get('negative', 0)*100:.1f}%</div>
            <div class="prob-card-label">Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_gauge_chart(confidence, prediction), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_bar_chart(probs), use_container_width=True)
    
    # Feature importance
    if "features" in result and result["features"]:
        st.markdown('<p class="section-header">🔍 Feature Importance</p>', unsafe_allow_html=True)
        st.plotly_chart(create_feature_chart(result["features"]), use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">Crypto Sentiment Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced MLOps-Powered Real-time Cryptocurrency Sentiment Analysis</p>', unsafe_allow_html=True)
    
    api = DashboardAPI()
    
    # Sidebar
    with st.sidebar:
        st.markdown("# Control Panel")
        st.markdown("<br>", unsafe_allow_html=True)
        
        health = api.get_health()
        
        if health.get("status") == "healthy":
            st.markdown('<div class="status-badge">🟢 API Online</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⏱️ Uptime", f"{health.get('uptime_seconds', 0)/3600:.1f}h")
            with col2:
                st.metric("💾 Memory", f"{health.get('memory_usage_mb', 0):.0f}MB")
        else:
            st.markdown('<div class="status-badge status-badge-offline">🔴 API Offline</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        model_info = api.get_model_info()
        if "error" not in model_info:
            st.markdown("### 🤖 Model Info")
            st.markdown(f"**Accuracy:** {model_info.get('accuracy', 0.846)*100:.1f}%")
            st.markdown(f"**Features:** {model_info.get('features_count', 43)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>84.6%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>156ms</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🎯 Real-time Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio("**Input Method:**", ["✍️ Manual Input", "📝 Sample Texts"], horizontal=True)
        
        if input_method == "✍️ Manual Input":
            text_input = st.text_area("**Enter text:**", placeholder="Bitcoin is great!", height=120)
        else:
            samples = {
                "🚀 Very Positive": "Bitcoin is mooning! This is incredible! 🚀💎",
                "😊 Positive": "Great news about Ethereum upgrade!",
                "😐 Neutral": "Bitcoin is trading at $125,000.",
                "😞 Negative": "Market is dumping hard today.",
                "💀 Very Negative": "Complete disaster! Lost everything!"
            }
            selected = st.selectbox("**Choose sample:**", list(samples.keys()))
            text_input = samples[selected]
        
        analyze_btn = st.button("🔮 ANALYZE SENTIMENT", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("- Enter crypto-related text\n- Longer = better analysis\n- Emojis work too! 😊")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results
    if analyze_btn and text_input:
        with st.spinner("🧠 Analyzing..."):
            result = api.predict_sentiment(text_input)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            elif "prediction" in result:
                show_results(result)

if __name__ == "__main__":
    main()