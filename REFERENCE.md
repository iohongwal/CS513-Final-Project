# StockML — Complete Project Reference
**CS513 · Stevens Institute of Technology · Spring 2026**
*Single source of truth for the team — last updated Apr 11, 2026*

---

## 1. Project Description

Binary classification of next-day stock price direction for AAPL, TSLA, NVDA,
and JPM. Predict whether tomorrow's close is **UP (≥ +0.5%)** or
**DOWN/FLAT** using five data streams and five ML algorithms. A controlled
dual-experiment design measures the added value of alternative data over
technical indicators alone. The best model is wrapped in an autonomous
four-step agentic system for live BUY/HOLD/SELL recommendations.

**Research Question:** Does alternative data (social media, search trends,
macro signals) meaningfully improve stock movement prediction accuracy beyond
technical indicators alone?

---

## 2. Data Pipeline

### 2.1 Data Streams

| Stream | Source | Library | Features Extracted |
|--------|--------|---------|-------------------|
| Price / Volume | Yahoo Finance | `yfinance` | RSI-14, MACD signal, Bollinger %B, MA5/MA20 ratio, Volume Z-score |
| Search Interest | Google Trends | `pytrends` | Trends Z-score, week-over-week Δ |
| Social Sentiment | Reddit r/WSB + VADER | `praw`, `vaderSentiment` | WSB mention count, mean compound sentiment |
| Market Context | SPY ETF | `yfinance` | Daily S&P 500 % return |
| Macroeconomic | FRED, CBOE VIX | `fredapi`, `yfinance` | 10Y–2Y yield spread, VIX close |

**Training window:** Jan 4, 2021 – Apr 9, 2026 (~4,800 rows, 4 tickers)

### 2.2 Label

```python
df["label"] = (df["close"].pct_change().shift(-1) > 0.005).astype(int)
# 1 = UP  (next-day return > +0.5%)
# 0 = DOWN / FLAT
```

### 2.3 Merge Logic

```python
df = tech.merge(trends, on="date", how="left")
         .merge(wsb,    on="date", how="left")
         .merge(spy,    on="date", how="left")
         .merge(macro,  on="date", how="left")
df.fillna({"wsb_count": 0, "wsb_sent": 0,
           "trends_z": 0, "trends_wow": 0}, inplace=True)
df.ffill(inplace=True)
df.dropna(inplace=True)
# Output → data/features/master_features.csv
```

### 2.4 Full Feature Set (12 Features)

| ID | Name | Stream | Description |
|----|------|--------|-------------|
| F1 | rsi_14 | Technical | Momentum oscillator (0–100) |
| F2 | macd_signal | Technical | 12/26/9 EMA trend direction |
| F3 | bb_pct | Technical | Price position in Bollinger Bands |
| F4 | ma_ratio | Technical | MA5 / MA20 cross signal |
| F5 | vol_z | Technical | 20-day rolling volume z-score |
| F6 | trends_z | Google Trends | Normalized daily search interest |
| F7 | trends_wow | Google Trends | Week-over-week search change |
| F8 | wsb_count | Reddit | Daily post mention count |
| F9 | wsb_sent | Reddit | Mean VADER compound score |
| F10 | spy_ret | Market | SPY ETF daily % return |
| F11 | vix | Macro | CBOE VIX daily close |
| F12 | yield_spread | Macro | 10Y − 2Y Treasury spread |

---

## 3. Methodology

### 3.1 Dual Experiment Design

| | Experiment A | Experiment B |
|---|---|---|
| Features | F1–F5 (technical only) | F1–F12 (all streams) |
| Role | Baseline (control) | Treatment |
| Key output | Accuracy, F1, ROC-AUC | ΔAccuracy = B − A |

### 3.2 Preprocessing Pipeline (per fold)

```
TimeSeriesSplit(n_splits=10)
    ├── Train fold
    │     └── ImbPipeline:
    │           ① MinMaxScaler   → scale all features to [0, 1]
    │           ② SMOTE          → balance classes (inside fold only)
    │           ③ Classifier     → fit
    └── Test fold
              └── scaler.transform(X_test) → evaluate
```

> SMOTE is applied only inside each training fold. Never on test data.

### 3.3 Five Classification Models

| # | Model | Config | Role |
|---|-------|--------|------|
| 1 | k-Nearest Neighbors | k=11, distance-weighted | Non-parametric baseline |
| 2 | Gaussian Naive Bayes | Default | Probabilistic lower bound |
| 3 | Decision Tree (CART) | max_depth=6, gini | Interpretable rule extraction |
| 4 | Random Forest | 100 trees, max_features=sqrt | Primary model + SHAP |
| 5 | ANN (MLPClassifier) | (64,32) ReLU, Adam | Non-linear interactions |

### 3.4 Evaluation Metrics

Accuracy · Precision · Recall · F1-score · ROC-AUC
*(reported per model per experiment)*

### 3.5 Extra Techniques

| Technique | Purpose | Where Applied |
|-----------|---------|--------------|
| SMOTE | Correct class imbalance (~55%/45%) | Inside each CV train fold |
| SHAP (TreeExplainer) | Rank feature importance by stream | Random Forest, Experiment B |

---

## 4. Agentic System

### 4.1 Four-Step Loop

```
STEP 1 — OBSERVE
  Fetch live data from all 5 streams for the target ticker (yfinance +
  pytrends + praw + FRED/VIX). Each stream wrapped in try/except;
  falls back to 0 if unavailable.

STEP 2 — THINK
  Recompute all 12 features (F1–F12) from live data using identical
  logic from the training pipeline. Apply stored MinMaxScaler.

STEP 3 — ACT
  Load rf_best.pkl → predict_proba(X_live)
  confidence = max(P(UP), P(DOWN))

STEP 4 — RECOMMEND
  BUY  → label=UP,   confidence ≥ 65%
  HOLD → label=UP,   confidence 55–65%
  SELL → label=DOWN, any confidence
  Output: signal + confidence % + top-3 SHAP feature drivers
```

### 4.2 Interfaces

```bash
# CLI
python agent/03_agent.py --ticker TSLA

# Streamlit dashboard (live demo)
streamlit run agent/streamlit_app.py
```

Streamlit panels: live price chart + RSI/MACD · social & macro signals ·
BUY/HOLD/SELL badge + confidence bar + SHAP explanation

---

## 5. Repository Structure

```
stockml-cs513/
├── README.md                      Full technical README
├── PROPOSAL.md                    1-page course proposal
├── REFERENCE.md                   This document
├── requirements.txt
├── data/
│   ├── raw/                       AAPL.csv, TSLA.csv, NVDA.csv, JPM.csv, SPY.csv
│   └── features/
│       └── master_features.csv    ← Person A delivers by Apr 13
├── src/
│   ├── 01_data_pipeline.py        Person A — fetch + merge + label
│   └── 02_train_models.py         Person B — train × 2 experiments + SHAP
├── agent/
│   ├── 03_agent.py                Person C — CLI agentic loop
│   └── streamlit_app.py           Person C — Streamlit live dashboard
├── models/
│   └── rf_best.pkl                ← Person B delivers by Apr 19
├── results/
│   ├── results_A.csv              Experiment A: 5 models × 5 metrics
│   ├── results_B.csv              Experiment B: 5 models × 5 metrics
│   └── figures/                   exp_a_vs_b.png, confusion_matrix.png, shap.png
└── slides/
    └── stockml_cs513.pptx         Person C delivers by Apr 26
```

---

## 6. Task Assignments

### Person A — Data Engineer
**Deadline: April 13**

- [ ] Set up repo, create folder structure, push `requirements.txt`
- [ ] Write `src/01_data_pipeline.py`:
  - [ ] Fetch OHLCV for AAPL, TSLA, NVDA, JPM via yfinance
  - [ ] Compute F1–F5 technical features using `ta` library
  - [ ] Fetch Google Trends for each ticker via pytrends (2×2.5yr chunks)
  - [ ] Fetch Reddit WSB posts via PRAW; score with VADER → F8, F9
  - [ ] Fetch SPY daily return → F10
  - [ ] Fetch VIX (^VIX) and FRED T10Y2Y → F11, F12
  - [ ] Merge all 5 streams by date (left join on trading calendar)
  - [ ] Create binary label (threshold ±0.5%)
  - [ ] Export to `data/features/master_features.csv`
- [ ] Write EDA notebook: class balance, correlation heatmap, outlier check
- [ ] Push `master_features.csv` to GitHub by Apr 13 ← critical handoff

**API keys needed:**
```python
REDDIT_CLIENT_ID     = "..."
REDDIT_CLIENT_SECRET = "..."
REDDIT_USER_AGENT    = "stockml/1.0"
FRED_API_KEY         = "..."
```

---

### Person B — ML Engineer
**Deadline: April 19 · Requires: master_features.csv from Person A**

- [ ] Write `src/02_train_models.py`:
  - [ ] Load `master_features.csv`
  - [ ] Define FEATURES_A (F1–F5) and FEATURES_B (F1–F12)
  - [ ] Build ImbPipeline: MinMaxScaler → SMOTE → Classifier
  - [ ] Train 5 models × 2 experiments using TimeSeriesSplit(n_splits=10)
  - [ ] Record Accuracy, Precision, Recall, F1, ROC-AUC per model per split
  - [ ] Save `results/results_A.csv` and `results/results_B.csv`
  - [ ] Generate `results/figures/exp_a_vs_b.png` (grouped bar chart)
  - [ ] Generate `results/figures/confusion_matrix_rf.png` (RF, Exp B)
  - [ ] Run SHAP TreeExplainer on RF (Exp B) → `shap_importance.png`
  - [ ] Save `models/rf_best.pkl` as dict: {model, scaler, features, metrics}
- [ ] Push `rf_best.pkl` to GitHub by Apr 19 ← critical handoff

**Recommended model configs:**
```python
models = {
  "knn":   KNeighborsClassifier(n_neighbors=11, weights="distance"),
  "gnb":   GaussianNB(),
  "dt":    DecisionTreeClassifier(max_depth=6, random_state=42),
  "rf":    RandomForestClassifier(n_estimators=100, random_state=42),
  "mlp":   MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500,
                         early_stopping=True, random_state=42)
}
```

---

### Person C — Agent Engineer + Presenter
**Deadline: April 26 · Requires: rf_best.pkl from Person B**

- [ ] Write `agent/03_agent.py`:
  - [ ] Load `rf_best.pkl` (model + scaler + feature list)
  - [ ] Implement OBSERVE step (all 5 streams, try/except per stream)
  - [ ] Implement THINK step (recompute F1–F12 from live data)
  - [ ] Implement ACT step (predict_proba → confidence %)
  - [ ] Implement RECOMMEND step (BUY/HOLD/SELL + SHAP explanation)
  - [ ] Add `--ticker` CLI argument (supports multiple tickers)
- [ ] Write `agent/streamlit_app.py`:
  - [ ] Panel 1: price line chart + RSI + MACD subplots
  - [ ] Panel 2: Trends bar + WSB sentiment gauge + SPY return metric
  - [ ] Panel 3: BUY/HOLD/SELL badge + confidence bar + top-3 SHAP features
- [ ] Build `slides/stockml_cs513.pptx` (11 slides — outline below)
- [ ] Rehearse live Streamlit demo (target: < 8 min for demo portion)

---

## 7. Presentation Outline (11 Slides)

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Title | Project name, team, course, date |
| 2 | Problem & Research Question | Binary task definition, motivation, ±0.5% threshold |
| 3 | Data Sources | 5-stream diagram with icons, ticker list |
| 4 | Feature Engineering | F1–F12 table colored by stream type |
| 5 | EDA | Class balance chart, correlation heatmap, top feature distributions |
| 6 | Methodology | Dual experiment design, TimeSeriesSplit diagram, SMOTE explanation |
| 7 | Results — Experiment A | Bar chart + table: 5 models × 5 metrics (technical only) |
| 8 | Results — Experiment B | Bar chart + table: 5 models × 5 metrics (all features) + ΔAccuracy |
| 9 | SHAP Feature Importance | Horizontal ranked bar chart by mean |SHAP|, colored by stream |
| 10 | Live Agent Demo ★ | Streamlit dashboard walkthrough — run live during presentation |
| 11 | Conclusions | Best model, ΔAccuracy finding, limitations, future work |

---

## 8. Team Guidelines

### Git Workflow
```bash
# Each person works on their own branch
git checkout -b feature/data-pipeline       # Person A
git checkout -b feature/model-training      # Person B
git checkout -b feature/agent-streamlit     # Person C

# Commit often with clear messages
git commit -m "feat: add RSI and MACD features to pipeline"
git commit -m "fix: prevent SMOTE leakage into test fold"

# Open a Pull Request into main when milestone is complete
# At least one teammate reviews before merge
```

### Branch → PR → Merge Rules
- Never push directly to `main`
- Every PR must include: what was done, output file paths, known issues
- `master_features.csv` and `rf_best.pkl` must be committed to `main` before
  the next person begins their branch

### Code Style
- All scripts must run end-to-end with `python src/01_data_pipeline.py` etc.
- Use `random_state=42` everywhere for reproducibility
- Wrap all external API calls in `try/except` with a printed warning
- No hard-coded file paths — use `pathlib.Path(__file__).parent`

### File Naming Conventions
```
data/raw/AAPL.csv           lowercase ticker
data/features/master_features.csv
models/rf_best.pkl
results/results_A.csv       experiment letter uppercase
results/figures/shap_importance.png
```

### Critical Handoff Dates
| File | From | To | By |
|------|------|----|----|
| `master_features.csv` | Person A | Person B | Apr 13 EOD |
| `rf_best.pkl` | Person B | Person C | Apr 19 EOD |
| All figures + slides | Person C | All | Apr 26 EOD |

---

## 9. API Setup Quick Reference

```python
# yfinance — no key needed
import yfinance as yf
df = yf.download("AAPL", start="2021-01-01", end="2026-04-09", auto_adjust=True)

# pytrends — no key needed
from pytrends.request import TrendReq
pt = TrendReq(); pt.build_payload(["AAPL"], timeframe="2021-01-01 2023-07-01")

# Reddit PRAW
import praw
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

# FRED
from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)
spread = fred.get_series("T10Y2Y", observation_start="2021-01-01")
```

---

## 10. Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| pytrends 429 rate limit | Pull in 2×2.5yr chunks; add `time.sleep(30)` between calls |
| Reddit PRAW historical limit | Use pushshift.io as fallback for pre-2023 posts if needed |
| FRED API downtime | Cache results locally after first successful pull |
| Class imbalance causing inflated accuracy | Use F1 + ROC-AUC as primary metrics, not accuracy alone |
| SMOTE data leakage | Use `imblearn.pipeline.Pipeline`, not standalone SMOTE before CV |
| rf_best.pkl sklearn version mismatch | Pin `scikit-learn==1.4.2` in requirements.txt; everyone uses same version |

---

*Educational purposes only. Not financial advice.*
