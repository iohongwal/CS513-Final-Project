import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure agent can be imported
import importlib
agent_module = importlib.import_module("agent.03_agent")
Agent = agent_module.Agent
from src.config import TICKERS

st.set_page_config(
    page_title="CS513 Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and dynamic animations
st.markdown("""
<style>
    .metric-value { font-size: 3rem; font-weight: 800; }
    .metric-label { font-size: 1.2rem; font-weight: 400; text-transform: uppercase; letter-spacing: 1px; color: #888;}
    .BUY { color: #00e676; }
    .HOLD { color: #ffb300; }
    .SELL { color: #ff1744; }
    .card { background-color: #1e1e1e; padding: 25px; border-radius: 12px; border: 1px solid #333; margin-bottom: 20px;}
    .driver-pos { color: #00e676; font-weight: bold;}
    .driver-neg { color: #ff1744; font-weight: bold;}
    
    /* Dynamic hover effects */
    div[data-testid="stSidebar"] {
        background-color: #121212 !important;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(110, 142, 251, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Autonomous Stock Recommender Agent")
st.markdown("*A CS513 Knowledge Discovery Final Project Demonstration*")

with st.sidebar:
    st.header("Control Panel")
    selected_ticker = st.selectbox("Select Target Ticker", TICKERS)
    analyze_btn = st.button(f"Analyze {selected_ticker} Live", use_container_width=True)
    st.markdown("---")
    st.caption("This agent runs real-time data ingestion across Yahoo, FRED, Reddit, and Google Trends to produce a predictive inference signal using our trained Random Forest.")

if analyze_btn:
    try:
        agent = Agent(selected_ticker)
        
        with st.status("Agent thinking...", expanded=True) as status:
            st.write("OBSERVE: Fetching multi-stream 60-day context...")
            live_row, price_history = agent.observe_and_think()
            
            st.write("ACT: Unbundling Random Forest & applying dynamic scaling...")
            output = agent.act_and_recommend(live_row)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
        
        # --- UI ROW 1: TOP METRICS ---
        st.markdown(f"### Current Status: {output['LiveDate']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Close Price</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${output["CurrentPrice"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Agent Recommendation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value {output["Action"]}">{output["Action"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{output["Confidence"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # --- UI ROW 2: SHAP & OHLC ---
        col_chart, col_shap = st.columns([2, 1])
        
        with col_chart:
            st.subheader("Recent Historical Context (60 Days)")
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            ax.plot(price_history.index, price_history['close'], color="#a777e3", linewidth=2)
            ax.grid(color='#333', linestyle='-', linewidth=0.5)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#333')
            st.pyplot(fig)
            
        with col_shap:
            st.subheader("Top Decision Drivers")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for feature, val in output["TopDrivers"]:
                cls = "driver-pos" if val > 0 else "driver-neg"
                icon = "▲" if val > 0 else "▼"
                st.markdown(f"**{feature.upper()}**")
                st.markdown(f"<span class='{cls}'>{icon} Impact: {val:.3f}</span>", unsafe_allow_html=True)
                st.progress(min(abs(val) / 0.1, 1.0)) # Hacky visualization of magnitude
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Agent Pipeline Failed: {str(e)}")
else:
    st.info("Select a ticker on the left sidebar and trigger the agent.")
