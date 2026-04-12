# CS513 StockML — Complete Codebase Documentation

> A file-by-file technical reference for every source file in the repository.  
> Generated for team review and final project submission.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Configuration & Environment](#2-configuration--environment)
   - 2.1 `.env` / `.env.example`
   - 2.2 `.gitignore`
   - 2.3 `requirements.txt`
   - 2.4 `src/config.py`
3. [Phase 1 — Data Pipeline](#3-phase-1--data-pipeline)
   - 3.1 `src/feature_utils.py`
   - 3.2 `src/01_data_pipeline.py`
4. [Phase 2 — Model Training & Evaluation](#4-phase-2--model-training--evaluation)
   - 4.1 `src/02_train_models.py`
5. [Phase 3 — Agent & Dashboard](#5-phase-3--agent--dashboard)
   - 5.1 `agent/03_agent.py`
   - 5.2 `agent/live_inference.py`
   - 5.3 `agent/streamlit_app.py`
6. [Generated Artifacts](#6-generated-artifacts)
7. [Data Flow Diagram](#7-data-flow-diagram)

---

## 1. Repository Structure

```
CS513-Final-Project/
├── .env                        # Secret API keys (never committed)
├── .env.example                # Template showing required env vars
├── .gitignore                  # Git exclusion rules
├── requirements.txt            # Python package dependencies
│
├── src/                        # Core backend logic (Phases 1 & 2)
│   ├── config.py               # Central configuration hub
│   ├── feature_utils.py        # Technical indicator math functions
│   ├── 01_data_pipeline.py     # Phase 1: Multi-source data scraper
│   └── 02_train_models.py      # Phase 2: ML training & evaluation
│
├── agent/                      # Phase 3: Autonomous agent layer
│   ├── 03_agent.py             # CLI ReAct agent
│   ├── live_inference.py       # Standalone inference utilities
│   └── streamlit_app.py        # Web dashboard frontend
│
├── data/
│   ├── raw/                    # Raw CSV per ticker (AAPL.csv, etc.)
│   └── features/
│       └── master_features.csv # Final unified feature table
│
├── models/
│   └── rf_best.pkl             # Serialized Random Forest bundle
│
└── results/
    ├── results_A.csv           # Experiment A metrics table
    ├── results_B.csv           # Experiment B metrics table
    └── figures/
        └── shap_summary.png    # Global SHAP bar chart
```

---

## 2. Configuration & Environment

### 2.1 `.env` / `.env.example`

**Purpose:** Store sensitive API credentials outside of source control.

| Variable | Service | What It Does |
|---|---|---|
| `REDDIT_CLIENT_ID` | Reddit (PRAW) | OAuth2 application ID from https://www.reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | Reddit (PRAW) | OAuth2 secret paired with the client ID |
| `REDDIT_USER_AGENT` | Reddit (PRAW) | Custom user-agent string (default: `stockml-cs513`) |
| `FRED_API_KEY` | Federal Reserve (FRED) | Personal key from https://fred.stlouisfed.org/docs/api/api_key.html |

`.env.example` serves as a **safe template** with empty values. Teammates clone the repo, copy it to `.env`, and fill in their own keys. Because `.env` is listed in `.gitignore`, actual secrets are never committed.

---

### 2.2 `.gitignore`

**Purpose:** Prevent generated and sensitive files from entering version control.

| Pattern | What It Excludes |
|---|---|
| `.venv/`, `venv/` | Python virtual environment |
| `.env` | Secret credentials file |
| `__pycache__/`, `*.pyc` | Python bytecode cache |
| `pip-*`, `matplotlib-*` | Temporary build/cache directories |
| `data/raw/`, `data/features/master_features.csv` | Reproducible data outputs |
| `results/*.csv`, `results/figures/*.png` | Reproducible experiment results |
| `models/*.pkl` | Serialized model files (too large for Git) |

**Design rationale:** All excluded files are *reproducible* by re-running the pipeline. This keeps the Git repository lightweight while ensuring no one accidentally pushes a 34 MB model file or leaks API keys.

---

### 2.3 `requirements.txt`

**Purpose:** Pin every Python dependency with version ranges for reproducibility.

The file is organized into logical sections:

| Section | Key Packages | Role |
|---|---|---|
| Data Collection | `yfinance`, `pytrends`, `praw`, `fredapi` | Pull data from Yahoo, Google Trends, Reddit, FRED |
| Feature Engineering | `ta`, `vaderSentiment` | Compute RSI/MACD/Bollinger and VADER sentiment scores |
| Data Handling | `pandas`, `numpy`, `scipy` | DataFrame manipulation and numerical math |
| Machine Learning | `scikit-learn`, `imbalanced-learn` | Classifiers, cross-validation, SMOTE oversampling |
| Explainability | `shap` | SHAP TreeExplainer for feature importance |
| Visualization | `matplotlib`, `seaborn` | Charts and plots |
| Agent / Dashboard | `streamlit` | Interactive web application framework |
| Utilities | `python-dotenv`, `joblib` | `.env` loading and model serialization |

**Notable version pins:**
- `pandas<3.0` and `numpy<2.0` — Both libraries have breaking API changes in their major releases.
- `scikit-learn>=1.4,<1.7` — Required by `imbalanced-learn` for pipeline compatibility.

---

### 2.4 `src/config.py`  
**Lines:** 64 | **Purpose:** Single source of truth for all project-wide constants.

This file is imported by *every other script* in the project. It performs three jobs:

#### Job 1: Path Resolution (Lines 9–17)
```python
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
```
Dynamically finds the project root regardless of the current working directory, then loads `.env` secrets into `os.environ`. All output paths (`DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `FIGURES_DIR`) are derived as children of `ROOT_DIR`.

#### Job 2: Feature Schema Definition (Lines 19–53)
Defines the exact columns used throughout the entire system:

| Constant | Contents | Used By |
|---|---|---|
| `TICKERS` | `["AAPL", "TSLA", "NVDA", "JPM", "AMZN", "MSFT", "META"]` | Pipeline, Agent |
| `TECHNICAL_FEATURES` | `rsi_14`, `macd_signal`, `bb_pct`, `ma_ratio`, `vol_z` | Experiment A |
| `ALT_FEATURES` | `trends_z`, `trends_wow`, `wsb_count`, `wsb_sent`, `spy_ret`, `vix`, `yield_spread` | Experiment B delta |
| `ALL_FEATURES` | `TECHNICAL_FEATURES + ALT_FEATURES` (12 total) | Experiment B, Agent |
| `SCHEMA_COLUMNS` | `[date, ticker] + ALL_FEATURES + [label]` | CSV column order |
| `TARGET_THRESHOLD` | `0.005` (0.5%) | Labels: if next-day return > 0.5% → UP (1) |

#### Job 3: Credential Loading (Lines 55–58)
Reads `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, and `FRED_API_KEY` from environment variables, defaulting to empty strings so the system degrades gracefully if keys are missing.

#### Utility Function: `ensure_directories()` (Lines 61–63)
Creates all output directories if they don't exist. Called at the start of every pipeline run to prevent `FileNotFoundError`.

---

## 3. Phase 1 — Data Pipeline

### 3.1 `src/feature_utils.py`  
**Lines:** 70 | **Purpose:** Pure mathematical functions for technical indicator computation.

This module has **zero side effects** — it takes DataFrames in and returns DataFrames out. No file I/O, no API calls.

#### `normalize_ohlcv(raw)` — Lines 16–31
- **Input:** Raw DataFrame from `yfinance` (may have MultiIndex columns, timezone-aware index).
- **Process:** Renames columns to lowercase (`Open` → `open`), strips timezone info, removes duplicate dates.
- **Output:** Clean DataFrame with columns `[open, high, low, close, volume]` indexed by date.
- **Why it exists:** `yfinance` returns inconsistent column formats across versions. This function normalizes everything into one predictable shape.

#### `add_technical_features(price_df)` — Lines 34–50
Computes the 5 technical indicators used in `TECHNICAL_FEATURES`:

| Feature | Formula | Library Call | Interpretation |
|---|---|---|---|
| `rsi_14` | 14-day Relative Strength Index | `ta.momentum.RSIIndicator` | >70 = overbought, <30 = oversold |
| `macd_signal` | MACD signal line | `ta.trend.MACD` | Crossover signals trend reversal |
| `bb_pct` | Bollinger Band %B | `ta.volatility.BollingerBands` | 0–1 range, >1 = above upper band |
| `ma_ratio` | 5-day SMA / 20-day SMA | `ta.trend.SMAIndicator` | >1 = short-term momentum bullish |
| `vol_z` | Volume z-score (20-day window) | Manual rolling calc | Measures volume anomalies |

#### `technical_snapshot(price_df)` — Lines 53–69
- Calls `add_technical_features()` and extracts only the **last row** as a dictionary.
- If any value is `NaN` (e.g., insufficient history for a 20-day window), it fills in sensible defaults (RSI → 50, ma_ratio → 1.0, etc.).
- **Used by:** `agent/live_inference.py` for real-time single-row feature extraction.

---

### 3.2 `src/01_data_pipeline.py`  
**Lines:** 356 | **Purpose:** The master data collection and assembly engine.

This is the largest and most complex file in the project. It orchestrates 5 external data sources into one unified CSV.

#### Conditional Imports (Lines 40–58)
Every external API library is wrapped in `try/except`:
```python
try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None
```
**Why:** If a teammate hasn't installed `praw` or doesn't have a `FRED_API_KEY`, the pipeline still runs — it simply fills those columns with zeros or medians instead of crashing. This is critical for development flexibility.

#### `fetch_price_history(ticker, start, end)` — Lines 88–106
- Calls `yf.download()` for daily OHLCV data.
- Normalizes via `normalize_ohlcv()`.
- Optionally saves raw CSV to `data/raw/{ticker}.csv`.
- **Raises `RuntimeError`** if Yahoo returns empty data (e.g., invalid ticker).

#### `fetch_spy_returns(start, end)` — Lines 109–112
- Downloads SPY (S&P 500 ETF) prices.
- Computes daily percentage returns via `.pct_change()`.
- Returns a single-column DataFrame `spy_ret` that gets joined to every ticker's features.
- **Purpose:** Captures broad market momentum as a feature — if the entire market drops, individual stock predictions should account for that.

#### `fetch_vix(start, end)` — Lines 115–117
- Downloads the CBOE Volatility Index (^VIX).
- Returns the raw closing value as the `vix` column.
- **Purpose:** High VIX = market fear/uncertainty; low VIX = market complacency. This is a direct alternative-data signal.

#### `fetch_yield_spread(start, end)` — Lines 120–143
- Uses the FRED API to download `DGS10` (10-Year Treasury) and `DGS2` (2-Year Treasury).
- Computes `yield_spread = DGS10 - DGS2`.
- Forward-fills missing days and reindexes to business days.
- **Graceful degradation:** If `FRED_API_KEY` is missing or the API call fails, returns `NaN` for every day (later filled by `finalize_dataset()`).
- **Purpose:** An inverted yield curve (negative spread) is a leading recession indicator.

#### `fetch_trends(ticker, start, end)` — Lines 159–205
- Uses `pytrends` to query Google Trends for `"{ticker} stock"`.
- Splits long date ranges into 900-day chunks (Google's API limit) via `_chunk_windows()`.
- Computes two features:
  - `trends_z` — Z-score of search interest relative to 20-day rolling mean (anomaly detection).
  - `trends_wow` — Week-over-week change (5-day diff).
- **Purpose:** Sudden spikes in Google searches for "TSLA stock" often precede volatility.

#### `fetch_reddit_features(ticker, start, end)` — Lines 208–259
- Authenticates to Reddit via `praw.Reddit()`.
- Scrapes up to 3,000 recent posts from `r/wallstreetbets`.
- Filters posts mentioning the ticker using regex `\bTSLA\b` (word-boundary match to avoid false positives).
- For each matching post, runs VADER sentiment analysis to get a compound score (−1 to +1).
- Aggregates per day into:
  - `wsb_count` — Number of mentions that day.
  - `wsb_sent` — Average sentiment of mentions that day.

#### `build_ticker_frame(ticker, start, end, spy_df, vix_df, spread_df)` — Lines 262–288
- Orchestrates the full per-ticker workflow: price → technicals → trends → reddit → market context.
- Joins all DataFrames on their date index using left-joins.
- **Creates the target label** (Line 283–284):
  ```python
  next_ret = frame["close"].shift(-1) / frame["close"] - 1.0
  frame["label"] = (next_ret > TARGET_THRESHOLD).astype(int)
  ```
  If tomorrow's close is >0.5% higher than today's close → label = 1 (UP), else label = 0 (DOWN).
- Drops the last row (which has no "next day" to compute a label for).

#### `finalize_dataset(df)` — Lines 291–310
- Sorts by date and ticker.
- Forward-fills NaN values within each ticker group (so Monday's missing Google Trends inherits Friday's value).
- Fills remaining NaN: social/trends columns → 0.0; vix/yield_spread → column median.
- Drops any rows still missing technical features (early rows without enough history for 20-day indicators).
- Enforces the exact `SCHEMA_COLUMNS` order and casts labels to `int`.

#### `run_pipeline(start, end, tickers)` — Lines 313–329
- The top-level orchestrator.
- Fetches shared market context (SPY, VIX, yield spread) once, then loops through each ticker.
- Concatenates all ticker frames, finalizes, and writes `master_features.csv`.

#### `main()` / CLI — Lines 340–355
- Supports command-line overrides: `--start`, `--end`, `--tickers`.
- Logs summary statistics (row count, date range, class balance).

---

## 4. Phase 2 — Model Training & Evaluation

### 4.1 `src/02_train_models.py`  
**Lines:** 192 | **Purpose:** Train, evaluate, and export the machine learning models.

#### Model Definitions (Lines 44–50)
Five classifiers with fixed hyperparameters:

| Name | Class | Key Parameters |
|---|---|---|
| kNN | `KNeighborsClassifier` | `n_neighbors=11, weights='distance'` |
| GNB | `GaussianNB` | (no hyperparameters) |
| CART | `DecisionTreeClassifier` | `max_depth=6, min_samples_leaf=20` |
| RandomForest | `RandomForestClassifier` | `n_estimators=100, max_features='sqrt'` |
| MLP | `MLPClassifier` | `hidden_layers=(64,32), alpha=0.001, max_iter=500` |

#### `build_pipeline(classifier)` — Lines 52–57
Wraps each classifier in an `imblearn.Pipeline`:
1. **`MinMaxScaler`** — Scales all features to [0, 1]. Essential for kNN and MLP which are distance-sensitive.
2. **`SMOTE`** — Synthetic Minority Oversampling applied only to each training fold (never the test fold). This addresses the ~60/40 class imbalance in our labels.
3. **Classifier** — The actual model.

**Why `imblearn.Pipeline` instead of `sklearn.Pipeline`?** Because `sklearn.Pipeline` doesn't support resamplers like SMOTE. The `imblearn` variant correctly applies SMOTE only during `.fit()`, never during `.predict()`.

#### `run_experiment(df, features)` — Lines 68–103
- Uses `TimeSeriesSplit(n_splits=10)` — this ensures training data is always chronologically before test data, preventing future-data leakage.
- For each of the 5 models × 10 folds = 50 evaluations:
  - Builds a fresh pipeline (to avoid state bleed between folds).
  - Records Accuracy, Precision, Recall, F1, and ROC-AUC.
- Averages fold metrics per model and returns a summary DataFrame.

#### `main()` — Lines 155–191
Executes the dual-experiment design:

1. **Experiment A:** `run_experiment(df, TECHNICAL_FEATURES)` — Only the 5 technical indicators.
2. **Experiment B:** `run_experiment(df, ALL_FEATURES)` — All 12 features including alternative data.
3. Prints a comparison table showing the accuracy delta per model.
4. Calls `train_and_explain_best_rf()` to export the final model.

#### `train_and_explain_best_rf(df, features)` — Lines 105–153
- Trains a final Random Forest on **all** available data (not cross-validated — this is the deployment model).
- Applies SMOTE to balance the full dataset before training.
- **SHAP Explainability** (Lines 120–141):
  - Subsamples 500 rows for computational efficiency.
  - Runs `shap.TreeExplainer` to decompose each prediction into per-feature contributions.
  - Generates a bar chart saved to `results/figures/shap_summary.png`.
- **Model Export** (Lines 143–153):
  - Packages `{'scaler': MinMaxScaler, 'model': RandomForest, 'features': list}` as a single dictionary.
  - Serialized via `joblib.dump()` to `models/rf_best.pkl` (~34 MB).
  - **Why a dictionary and not the imblearn pipeline?** At inference time, we must NOT apply SMOTE to incoming data. Bundling the scaler and model separately gives the agent clean control.

---

## 5. Phase 3 — Agent & Dashboard

### 5.1 `agent/03_agent.py`  
**Lines:** 154 | **Purpose:** Command-line autonomous agent implementing the ReAct loop.

#### `class Agent` — Lines 32–144

**`__init__(self, ticker)`** — Lines 33–48
- Validates the ticker against the configured list (warns but doesn't block unknown tickers).
- Loads the `rf_best.pkl` bundle and unpacks scaler, model, and feature list.
- Sets a 60-day lookback window from today — this provides enough history for the 20-day rolling indicators.

**`observe_and_think(self)`** — Lines 50–80 (ReAct Steps 1 & 2)
- **OBSERVE:** Fires off 6 real-time API calls in sequence:
  1. `fetch_spy_returns()` — Today's market return
  2. `fetch_vix()` — Current volatility index
  3. `fetch_yield_spread()` — Treasury curve from FRED
  4. `fetch_price_history()` — 60 days of OHLCV for the target ticker
  5. `fetch_trends()` — Google Trends z-score
  6. `fetch_reddit_features()` — r/wallstreetbets sentiment
- **THINK:** Joins all 6 DataFrames on the date index, fills NaN values, drops incomplete rows, and isolates the **last row** as the "live" feature vector.
- Returns both the live feature row and the raw price history (for charting).

**`act_and_recommend(self, live_row)`** — Lines 82–126 (ReAct Steps 3 & 4)
- **ACT:** Extracts the 12 features, applies `scaler.transform()`, and calls `model.predict_proba()`.
- **RECOMMEND:** Maps the UP probability to a decision:
  - ≥ 65% → **BUY**
  - 55–65% → **HOLD**
  - < 55% → **SELL** (confidence = DOWN probability)
- **SHAP local explanation:** Runs `TreeExplainer.shap_values()` on the single scaled row, sorts features by absolute SHAP impact, and returns the top 3 as "drivers."

**`execute_loop(self)`** — Lines 128–144
- Pretty-prints the full output to stdout for terminal use.

**CLI entry point** — Lines 146–153
```bash
python agent/03_agent.py --ticker TSLA
```

---

### 5.2 `agent/live_inference.py`  
**Lines:** 229 | **Purpose:** A lower-level, function-based inference toolkit.

This file provides an **alternative** path to the `Agent` class and was written before `03_agent.py`. It is modular and stateless — each function takes explicit inputs and returns outputs.

#### Key Functions

| Function | Lines | What It Does |
|---|---|---|
| `load_bundle()` | 65–68 | Loads `rf_best.pkl` and returns the raw dictionary |
| `_latest_trends_values(ticker)` | 71–98 | Fetches 3-month Google Trends, computes z-score and wow for the latest day only |
| `_latest_reddit_values(ticker)` | 101–129 | Scrapes 400 recent r/wsb posts, counts mentions and averages VADER sentiment |
| `_latest_market_context()` | 132–165 | Gets SPY return, VIX close, and yield spread for the most recent trading day |
| `compute_live_features(ticker, features)` | 168–181 | Orchestrates all of the above into a single-row DataFrame matching the model's expected schema |
| `recommendation_label(prob_up)` | 184–189 | Maps probability to BUY/HOLD/SELL string |
| `top_feature_drivers(bundle, scaled_row)` | 192–211 | SHAP-based explanation with fallback to `feature_importances_` if SHAP fails |
| `predict_ticker(ticker, bundle)` | 214–228 | Full end-to-end: features → scale → predict → explain → return dict |

**Difference from `03_agent.py`:** This module downloads only 90 days of prices via `yf.download(..., period="90d")` and computes a single snapshot.  The `Agent` class in `03_agent.py` uses explicit date ranges and returns the full price history for charting.

---

### 5.3 `agent/streamlit_app.py`  
**Lines:** 127 | **Purpose:** Interactive web dashboard for live presentations.

#### Page Configuration (Lines 17–22)
- Sets page title `"CS513 Agent Dashboard"`, robot emoji icon, wide layout, and expanded sidebar.

#### Custom CSS (Lines 24–53)
Injects dark-mode styling directly into the page:
- `.BUY` → green (#00e676), `.SELL` → red (#ff1744), `.HOLD` → amber (#ffb300)
- Dark card backgrounds (`#1e1e1e`) with subtle borders
- Gradient-styled analyze button with hover lift animation
- Dark sidebar background

#### Sidebar Controls (Lines 58–63)
- A `st.selectbox` dropdown populated from `config.TICKERS` (currently 7 stocks).
- A gradient "Analyze {ticker} Live" button.

#### Analysis Flow (Lines 65–124)
When the button is clicked:

1. **Spinner UX** — `st.status("Agent thinking...")` shows real-time progress messages while the agent scrapes data (~15 seconds).
2. **Metric Cards** — Three columns display:
   - Close Price (formatted as USD)
   - Recommendation (color-coded BUY/HOLD/SELL)
   - Confidence percentage
3. **Price Chart** — Matplotlib line chart of the 60-day closing prices, styled with dark background matching the UI theme.
4. **SHAP Drivers** — For each of the top-3 features:
   - Feature name in bold
   - Green ▲ or Red ▼ with impact value
   - A `st.progress()` bar showing relative magnitude

#### Error Handling (Lines 123–126)
If any part of the agent pipeline fails, `st.error()` displays the exception message instead of crashing the entire app.

---

## 6. Generated Artifacts

These files are produced by running the pipeline and are **not committed to Git**:

| File | Generated By | Size | Contents |
|---|---|---|---|
| `data/raw/{TICKER}.csv` | `01_data_pipeline.py` | ~50 KB each | Raw OHLCV per ticker |
| `data/features/master_features.csv` | `01_data_pipeline.py` | ~942 KB | 9,023 rows × 15 columns |
| `results/results_A.csv` | `02_train_models.py` | ~0.5 KB | 5 models × 5 metrics (tech only) |
| `results/results_B.csv` | `02_train_models.py` | ~0.5 KB | 5 models × 5 metrics (all features) |
| `results/figures/shap_summary.png` | `02_train_models.py` | ~124 KB | Global SHAP feature importance bar chart |
| `models/rf_best.pkl` | `02_train_models.py` | ~34 MB | Serialized `{scaler, model, features}` dict |

---

## 7. Data Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │            External APIs                │
                    │  Yahoo  · Google · Reddit · FRED        │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────┐
              │         01_data_pipeline.py               │
              │  fetch_price   → add_technical_features   │
              │  fetch_trends  → trends_z, trends_wow     │
              │  fetch_reddit  → wsb_count, wsb_sent      │
              │  fetch_spy/vix/yield                       │
              │  build_ticker_frame → finalize_dataset     │
              └─────────────────┬────────────────────────┘
                                │
                  master_features.csv  (9,023 × 15)
                                │
                                ▼
              ┌──────────────────────────────────────────┐
              │         02_train_models.py                │
              │  Exp A: TECHNICAL_FEATURES (5 cols)       │
              │  Exp B: ALL_FEATURES (12 cols)            │
              │  TimeSeriesSplit → MinMaxScaler → SMOTE   │
              │  → kNN / NB / CART / RF / MLP             │
              │  SHAP TreeExplainer → shap_summary.png    │
              └─────────────────┬────────────────────────┘
                                │
                  rf_best.pkl  {scaler, model, features}
                                │
                                ▼
              ┌──────────────────────────────────────────┐
              │         03_agent.py / streamlit_app.py    │
              │  OBSERVE: fetch 60-day live data          │
              │  THINK:   compute 12 features             │
              │  ACT:     scaler.transform → predict      │
              │  RECOMMEND: BUY / HOLD / SELL + SHAP      │
              └──────────────────────────────────────────┘
```
