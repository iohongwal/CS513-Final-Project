from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from live_inference import load_bundle, predict_ticker

st.set_page_config(page_title="Stock Agent Dashboard", layout="wide")
st.title("Stock Movement Agent")
st.caption("Observe -> Think -> Act -> Recommend")

model_path = st.text_input("Model path", value=str(Path("models") / "rf_best.pkl"))

try:
    bundle = load_bundle(Path(model_path))
except Exception as exc:
    st.error(f"Failed to load model bundle: {exc}")
    st.stop()

left, right = st.columns([1, 2])
with left:
    ticker = st.selectbox("Ticker", ["AAPL", "TSLA", "NVDA", "JPM"], index=0)
    run = st.button("Run Prediction", type="primary")

if run:
    try:
        result = predict_ticker(ticker, bundle)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    with right:
        st.subheader(f"Recommendation: {result['recommendation']}")
        st.metric("Probability UP", f"{result['prob_up']:.2%}")
        st.progress(max(0.0, min(1.0, result["prob_up"])))

        st.markdown("### Top-3 Drivers")
        drivers_df = pd.DataFrame(result["drivers"], columns=["feature", "impact"])
        st.dataframe(drivers_df, use_container_width=True, hide_index=True)

    st.markdown("### Feature Snapshot")
    st.dataframe(result["feature_row"], use_container_width=True, hide_index=True)
else:
    st.info("Select a ticker and click Run Prediction.")
