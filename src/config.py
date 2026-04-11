from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEATURE_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

START_DATE = "2021-01-04"
END_DATE = date.today().isoformat()

TICKERS = ["AAPL", "TSLA", "NVDA", "JPM"]
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
TARGET_COLUMN = "label"
TARGET_THRESHOLD = 0.005

TECHNICAL_FEATURES = [
    "rsi_14",
    "macd_signal",
    "bb_pct",
    "ma_ratio",
    "vol_z",
]

ALT_FEATURES = [
    "trends_z",
    "trends_wow",
    "wsb_count",
    "wsb_sent",
    "spy_ret",
    "vix",
    "yield_spread",
]

ALL_FEATURES = TECHNICAL_FEATURES + ALT_FEATURES
SCHEMA_COLUMNS = [DATE_COLUMN, TICKER_COLUMN] + ALL_FEATURES + [TARGET_COLUMN]

MASTER_FEATURES_PATH = FEATURE_DATA_DIR / "master_features.csv"
RF_BEST_MODEL_PATH = MODELS_DIR / "rf_best.pkl"

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "stockml-cs513")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def ensure_directories() -> None:
    for path in [RAW_DATA_DIR, FEATURE_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
