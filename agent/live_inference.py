from __future__ import annotations

import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    FRED_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    RF_BEST_MODEL_PATH,
    SPY_TICKER,
    VIX_TICKER,
)
from src.feature_utils import normalize_ohlcv, technical_snapshot  # noqa: E402

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    import praw
except Exception:
    praw = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from fredapi import Fred
except Exception:
    Fred = None


def _configure_yfinance_cache() -> None:
    try:
        cache_dir = ROOT_DIR / ".cache" / "py-yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        return


_configure_yfinance_cache()


def load_bundle(model_path: Path = RF_BEST_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    return joblib.load(model_path)


def _latest_trends_values(ticker: str) -> dict[str, float]:
    if TrendReq is None:
        return {"trends_z": 0.0, "trends_wow": 0.0}

    keyword = f"{ticker} stock"
    try:
        client = TrendReq(hl="en-US", tz=360)
        client.build_payload([keyword], timeframe="today 3-m")
        frame = client.interest_over_time()

        value_cols = [col for col in frame.columns if col != "isPartial"]
        if frame.empty or not value_cols:
            return {"trends_z": 0.0, "trends_wow": 0.0}

        ser = frame[value_cols[0]].astype(float)
        latest = ser.iloc[-1]
        roll_mean = ser.rolling(20).mean().iloc[-1]
        roll_std = ser.rolling(20).std().iloc[-1]

        z = (latest - roll_mean) / roll_std if pd.notna(roll_std) and roll_std != 0 else 0.0
        wow = latest - ser.iloc[-6] if len(ser) >= 6 else 0.0

        return {
            "trends_z": float(np.nan_to_num(z, nan=0.0)),
            "trends_wow": float(np.nan_to_num(wow, nan=0.0)),
        }
    except Exception:
        return {"trends_z": 0.0, "trends_wow": 0.0}


def _latest_reddit_values(ticker: str) -> dict[str, float]:
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or praw is None or SentimentIntensityAnalyzer is None:
        return {"wsb_count": 0.0, "wsb_sent": 0.0}

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False,
        )
        analyzer = SentimentIntensityAnalyzer()
        pattern = re.compile(rf"\b{re.escape(ticker)}\b", flags=re.IGNORECASE)

        mentions = 0
        scores: list[float] = []

        for post in reddit.subreddit("wallstreetbets").new(limit=400):
            text = f"{post.title} {post.selftext or ''}"
            if pattern.search(text):
                mentions += 1
                scores.append(analyzer.polarity_scores(text)["compound"])

        return {
            "wsb_count": float(mentions),
            "wsb_sent": float(np.mean(scores)) if scores else 0.0,
        }
    except Exception:
        return {"wsb_count": 0.0, "wsb_sent": 0.0}


def _latest_market_context() -> dict[str, float]:
    try:
        spy = yf.download(SPY_TICKER, period="10d", interval="1d", progress=False, auto_adjust=False, threads=False)
        spy_close = normalize_ohlcv(spy)["close"]
        spy_ret = float(spy_close.pct_change().iloc[-1]) if len(spy_close) > 1 else 0.0
    except Exception:
        spy_ret = 0.0

    try:
        vix = yf.download(VIX_TICKER, period="10d", interval="1d", progress=False, auto_adjust=False, threads=False)
        vix_val = float(normalize_ohlcv(vix)["close"].iloc[-1])
    except Exception:
        vix_val = 20.0

    yield_spread = 0.0
    if Fred is not None and FRED_API_KEY:
        try:
            now = pd.Timestamp.utcnow().tz_localize(None)
            start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

            fred = Fred(api_key=FRED_API_KEY)
            ten = pd.to_numeric(fred.get_series("DGS10", observation_start=start, observation_end=end), errors="coerce")
            two = pd.to_numeric(fred.get_series("DGS2", observation_start=start, observation_end=end), errors="coerce")
            spread = (ten - two).dropna()
            yield_spread = float(spread.iloc[-1]) if not spread.empty else 0.0
        except Exception:
            yield_spread = 0.0

    return {
        "spy_ret": float(np.nan_to_num(spy_ret, nan=0.0)),
        "vix": float(np.nan_to_num(vix_val, nan=20.0)),
        "yield_spread": float(np.nan_to_num(yield_spread, nan=0.0)),
    }


def compute_live_features(ticker: str, expected_features: list[str]) -> pd.DataFrame:
    market = yf.download(ticker, period="90d", interval="1d", progress=False, auto_adjust=False, threads=False)
    if market.empty:
        raise RuntimeError(f"No market data available for {ticker}")

    price = normalize_ohlcv(market)
    features: dict[str, float] = {}
    features.update(technical_snapshot(price))
    features.update(_latest_trends_values(ticker))
    features.update(_latest_reddit_values(ticker))
    features.update(_latest_market_context())

    row = {name: float(np.nan_to_num(features.get(name, 0.0), nan=0.0)) for name in expected_features}
    return pd.DataFrame([row])


def recommendation_label(prob_up: float) -> str:
    if prob_up >= 0.65:
        return "BUY"
    if prob_up >= 0.55:
        return "HOLD"
    return "SELL"


def top_feature_drivers(bundle: dict[str, Any], scaled_row: np.ndarray, top_n: int = 3) -> list[tuple[str, float]]:
    model = bundle["model"]
    features = bundle["features"]

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(scaled_row)
        if isinstance(values, list):
            vector = values[1][0]
        else:
            vector = values[0]

        order = np.argsort(np.abs(vector))[::-1][:top_n]
        return [(features[i], float(vector[i])) for i in order]
    except Exception:
        importances = getattr(model, "feature_importances_", np.zeros(len(features)))
        order = np.argsort(importances)[::-1][:top_n]
        return [(features[i], float(importances[i])) for i in order]


def predict_ticker(ticker: str, bundle: dict[str, Any]) -> dict[str, Any]:
    row = compute_live_features(ticker=ticker, expected_features=bundle["features"])
    scaled = bundle["scaler"].transform(row)

    prob_up = float(bundle["model"].predict_proba(scaled)[:, 1][0])
    drivers = top_feature_drivers(bundle, scaled)

    return {
        "ticker": ticker,
        "prob_up": prob_up,
        "recommendation": recommendation_label(prob_up),
        "drivers": drivers,
        "feature_row": row,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
