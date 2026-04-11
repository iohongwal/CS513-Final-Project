# Stock Movement Classification with an Agentic Recommendation Component
**CS513 Knowledge Discovery & Data Mining · Stevens Institute of Technology · Spring 2026**

---

## Overview

We propose a supervised binary classification system that predicts whether a
stock's next-day closing price will move **UP (≥ +0.5%)** or **DOWN/FLAT**
for four equities — AAPL, TSLA, NVDA, and JPM — over a five-year window
(January 2021 – April 2026). The system is extended by an autonomous agentic
pipeline that retrieves live data and produces interpretable BUY/HOLD/SELL
recommendations in real time.

---

## Research Question

> Does incorporating alternative data — social media sentiment, internet search
> trends, and macroeconomic signals — improve binary stock movement prediction
> accuracy beyond technical indicators alone?

---

## Data & Features

Five data streams are merged into a unified daily feature matrix (~4,800 rows):

| Stream | Source | Features |
|--------|--------|----------|
| Price & Volume | yfinance | RSI-14, MACD, Bollinger %B, MA5/MA20 ratio, Volume Z-score |
| Search Interest | Google Trends (pytrends) | Search volume Z-score, week-over-week change |
| Social Sentiment | Reddit WSB + VADER | Daily mention count, mean compound sentiment |
| Market Context | SPY ETF (yfinance) | S&P 500 daily % return |
| Macroeconomic | FRED API + CBOE VIX | VIX close, 10Y–2Y yield spread |

**Label:** `1 (UP)` if next-day return > +0.5%, else `0 (DOWN/FLAT)`

---

## Methodology

A **dual-experiment design** isolates the value of alternative data:

- **Experiment A (Baseline):** 5 technical features only
- **Experiment B (Treatment):** All 12 features including alternative streams

Both experiments apply the same pipeline across **5 classification models**:

| Model | Role |
|-------|------|
| k-Nearest Neighbors | Non-parametric baseline |
| Gaussian Naive Bayes | Probabilistic reference |
| Decision Tree (CART) | Interpretable rule extraction |
| Random Forest | Primary model; expected best performer |
| ANN / MLPClassifier | Non-linear interaction capture |

**Preprocessing:** `MinMaxScaler → SMOTE → Classifier` inside `TimeSeriesSplit
(n_splits=10)` — preserving temporal order and preventing data leakage.

**Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC per model per
experiment. ΔAccuracy = Experiment B − Experiment A quantifies alternative
data contribution. **SHAP** (TreeExplainer) ranks feature importance.

---

## Agentic Component *(Extra Credit)*

The best-performing model is embedded in a four-step autonomous agent:

```
OBSERVE → fetch live data from all 5 streams
THINK   → recompute all 12 features
ACT     → load rf_best.pkl → predict label + confidence %
RECOMMEND → BUY (conf ≥ 65%) · HOLD (55–65%) · SELL (DOWN)
            + top-3 SHAP feature drivers printed per call
```

Deployed via CLI (`python agent.py --ticker TSLA`) and a **Streamlit
dashboard** for live class demonstration.

---

## Timeline

| Dates | Deliverable | Owner |
|-------|-------------|-------|
| Apr 10–13 | Data pipeline → `master_features.csv` | Person A |
| Apr 14–19 | 5 models × 2 experiments + SHAP + `rf_best.pkl` | Person B |
| Apr 21–26 | Agentic CLI + Streamlit app + slide deck | Person C |
| Apr 28 | Full rehearsal | All |
| Apr 30 | CANVAS submission + live presentation | All |

---

*Disclaimer: Educational purposes only. Not financial advice.*
