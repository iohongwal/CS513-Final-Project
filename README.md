# Stock Movement Classification with an Agentic Recommendation Component

**CS513 Knowledge Discovery & Data Mining**  
Stevens Institute of Technology · School of Engineering and Science  
Spring 2026 · Group Project Proposal

---

## Abstract

This project proposes a supervised binary classification system to predict
next-day stock price movement — defined as UP (≥ +0.5%) or DOWN/FLAT —
for four publicly traded equities: AAPL, TSLA, NVDA, and JPM. The system
integrates five heterogeneous data streams spanning technical market signals,
internet search behavior, social media sentiment, broad market context, and
macroeconomic indicators. A controlled dual-experiment design isolates the
incremental predictive value of alternative data over technical indicators
alone across five classification algorithms taught in CS513. The best-performing
model is embedded in an autonomous agentic pipeline that retrieves live data,
recomputes features, and produces an actionable BUY / HOLD / SELL
recommendation with confidence score and feature-level explanation.

---

## 1. Motivation and Problem Statement

Predicting stock price movement is a canonical problem in financial machine
learning with direct practical significance. Classical approaches rely on
technical indicators derived solely from historical price and volume. However,
modern data availability presents a richer landscape: retail investor behavior
is increasingly observable through social media; macroeconomic uncertainty is
quantifiable through volatility indices and yield curves; and public interest
in individual equities is measurable through search engine query volume.

This project investigates whether combining these heterogeneous signals into a
unified feature matrix meaningfully improves binary classification accuracy
relative to a technical-only baseline — and whether the resulting model can
be operationalized as a real-time agentic decision system.

**Research Question:**
> Does incorporating alternative data — social media sentiment, internet search
> trends, and macroeconomic signals — improve binary stock movement prediction
> accuracy beyond what technical indicators alone can achieve?

---

## 2. Objectives

1. Collect and integrate five distinct real-world data streams for four equity
   tickers over a five-year training window (2021–2026).
2. Engineer 12 quantitative features and a binary classification label from
   the merged dataset.
3. Train and evaluate five classification models using time-series-aware
   cross-validation, reporting five performance metrics per model per experiment.
4. Quantify the marginal contribution of alternative data via a controlled
   A/B experiment (technical-only vs. full feature set).
5. Explain model behavior using SHAP (SHapley Additive exPlanations) to
   identify the most informative features per prediction.
6. Deploy the best-performing model within an autonomous four-step agentic
   pipeline that produces live, interpretable investment signals.

---

## 3. Dataset

### 3.1 Target Securities

| Ticker | Company | Sector | Rationale |
|--------|---------|--------|-----------|
| AAPL | Apple Inc. | Technology | Large-cap, high liquidity, broad retail investor following |
| TSLA | Tesla Inc. | Consumer Discretionary | High social media sensitivity, retail-driven volatility |
| NVDA | NVIDIA Corp. | Semiconductors | Strong search trend signal tied to AI narrative cycles |
| JPM | JPMorgan Chase | Financials | Macro-sensitive; captures yield curve and VIX effects |

### 3.2 Training Window

| Parameter | Value |
|-----------|-------|
| Start date | January 4, 2021 |
| End date | April 9, 2026 |
| Trading days per ticker | ~1,250 |
| Total dataset size | ~4,800 rows × 14 columns |

### 3.3 Data Streams

| Stream | Source | Library | Update Frequency |
|--------|--------|---------|-----------------|
| Price & Volume (OHLCV) | Yahoo Finance | `yfinance` | Daily |
| Search Interest | Google Trends | `pytrends` | Weekly → forward-filled daily |
| Social Media Posts | Reddit r/wallstreetbets | `praw`, `vaderSentiment` | Daily aggregated |
| Broad Market Return | S&P 500 ETF (SPY) | `yfinance` | Daily |
| Macroeconomic Signals | FRED API, CBOE (^VIX) | `fredapi`, `yfinance` | Daily |

### 3.4 Label Definition

```python
# Binary target: 1 = price moves UP by more than 0.5% next trading day
df["label"] = (df["close"].pct_change().shift(-1) > 0.005).astype(int)
```

A threshold of ±0.5% is applied to reduce label noise from microstructure
fluctuations, consistent with approaches reported in the empirical finance
literature.

---

## 4. Feature Engineering

### 4.1 Complete Feature Matrix

| ID | Feature | Data Stream | Description |
|----|---------|-------------|-------------|
| F1 | RSI-14 | Technical | Relative Strength Index over 14 days (0–100) |
| F2 | MACD Signal | Technical | Difference of 12-day and 26-day EMA, smoothed by 9-day EMA |
| F3 | Bollinger %B | Technical | Price position relative to upper and lower Bollinger Bands |
| F4 | MA5/MA20 Ratio | Technical | 5-day SMA divided by 20-day SMA |
| F5 | Volume Z-score | Technical | Standardized 20-day rolling volume deviation |
| F6 | Trends Z-score | Google Trends | Normalized daily search interest per ticker |
| F7 | Trends WoW Δ | Google Trends | Week-over-week percentage change in search volume |
| F8 | WSB Count | Reddit | Daily count of r/wallstreetbets posts mentioning ticker |
| F9 | WSB Sentiment | Reddit + VADER | Mean VADER compound sentiment score across daily posts |
| F10 | SPY Return | Market Index | SPY ETF daily percentage return |
| F11 | VIX | Macroeconomic | CBOE Volatility Index daily close |
| F12 | Yield Spread | Macroeconomic | 10-year minus 2-year U.S. Treasury yield (FRED: T10Y2Y) |

### 4.2 Feature Categorization

- **Experiment A features:** F1–F5 (technical indicators only)
- **Experiment B features:** F1–F12 (full alternative + technical feature set)

---

## 5. Methodology

### 5.1 Preprocessing Pipeline

All five models share an identical preprocessing pipeline implemented via
`imblearn.pipeline.Pipeline` to ensure consistent, leakage-free evaluation:

```
Input features
    → MinMaxScaler          (normalize all features to [0, 1])
    → SMOTE                 (balance classes within each training fold only)
    → Classifier
```

SMOTE (Synthetic Minority Over-sampling Technique) is applied exclusively
inside each cross-validation training fold. It is never applied to the test
fold, ensuring that evaluation metrics reflect real-world generalization.

### 5.2 Cross-Validation Strategy

Given the temporal nature of financial data, standard k-fold cross-validation
is not appropriate because it allows future information to leak into training
sets. We use `sklearn.model_selection.TimeSeriesSplit(n_splits=10)`, which
expands the training window chronologically across ten folds.

```
Fold 1:  Train [days 1–480]    → Test [days 481–528]
Fold 2:  Train [days 1–528]    → Test [days 529–576]
...
Fold 10: Train [days 1–4,320]  → Test [days 4,321–4,800]
```

### 5.3 Experiment Design

| | Experiment A | Experiment B |
|---|---|---|
| **Role** | Baseline (control) | Treatment |
| **Features** | F1–F5 only | F1–F12 |
| **Purpose** | Establish technical-only benchmark | Quantify contribution of alternative data |
| **Key output** | Accuracy, F1, ROC-AUC per model | ΔAccuracy = B − A per model |

### 5.4 Classification Models

#### Model 1 — k-Nearest Neighbors (kNN)
Identifies trading days with similar feature profiles and predicts based on
their outcomes. Appropriate as a non-parametric baseline that captures local
structure in feature space. Configuration: k=11 (odd to avoid ties),
distance-weighted voting, Euclidean metric.

#### Model 2 — Gaussian Naive Bayes (GNB)
A probabilistic classifier assuming Gaussian-distributed, conditionally
independent features. Serves as a fast probabilistic lower bound. Most likely
to underperform due to correlations among technical indicators, but provides
a useful reference point and clear interpretability of class likelihoods.

#### Model 3 — Decision Tree (CART)
Partitions the feature space using axis-aligned thresholds selected to minimize
Gini impurity. Provides fully interpretable IF-THEN rules directly visualizable
in the project presentation. Configuration: max_depth=6, min_samples_leaf=20.

#### Model 4 — Random Forest
An ensemble of 100 decorrelated decision trees trained on bootstrap samples,
with random feature subsets at each split (max_features = √12 ≈ 3). Reduces
variance compared to a single tree and handles non-linear feature interactions.
Expected to achieve the highest or second-highest accuracy. Selected as the
deployed model for the agentic pipeline.

#### Model 5 — Multilayer Perceptron (ANN)
A feedforward neural network with two hidden layers (64 → 32 units, ReLU
activation, Adam optimizer, L2 regularization α=0.001). Captures complex
non-linear interactions across all 12 features simultaneously. Evaluated as
a representative deep learning approach within the scope of CS513 Module 11.

### 5.5 Evaluation Metrics

Each model is evaluated on five metrics across both experiments:

| Metric | Interpretation in Context |
|--------|--------------------------|
| Accuracy | Overall fraction of correct UP/DOWN predictions |
| Precision | Of days predicted UP, the fraction that truly moved UP |
| Recall | Of days that truly moved UP, the fraction correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Discrimination ability across all classification thresholds |

### 5.6 Extra Techniques

**SMOTE** (Chawla et al., 2002): Addresses natural class imbalance (~55% UP,
~45% DOWN) by synthesizing minority-class samples via k-NN interpolation in
feature space, applied strictly within training folds.

**SHAP** (Lundberg & Lee, 2017): TreeExplainer computes Shapley values for
each feature of each prediction from the Random Forest. Mean absolute SHAP
values are ranked to produce a global feature importance chart stratified by
data stream type.

---

## 6. Agentic System

### 6.1 Design Rationale

The agentic component extends the project from a static offline experiment
into a prototype of an autonomous financial decision-support system. Rather
than requiring manual data collection and feature computation at inference
time, the agent independently executes the full observe-to-recommend cycle
on any supported ticker with a single command.

The design follows the ReAct (Reason + Act) framework, where an agent
alternates between gathering environmental information and taking goal-directed
actions without human intervention.

### 6.2 Four-Step Agent Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  STEP 1 — OBSERVE                                           │
│  Fetch fresh data from all five streams for the target      │
│  ticker on the current trading day. Each stream is wrapped  │
│  in a try/except block; unavailable streams fall back to    │
│  zero without halting the pipeline.                         │
│                                                             │
│  STEP 2 — THINK                                             │
│  Recompute all 12 features (F1–F12) from the live data      │
│  using the identical logic applied during model training.   │
│  Ensures input distribution consistency between train and   │
│  inference.                                                 │
│                                                             │
│  STEP 3 — ACT                                               │
│  Load rf_best.pkl (bundled model + scaler + feature list).  │
│  Apply the stored MinMaxScaler to the live feature vector.  │
│  Call predict_proba() to obtain class probabilities.        │
│  Compute confidence = max(P(UP), P(DOWN)).                  │
│                                                             │
│  STEP 4 — RECOMMEND                                         │
│  Map prediction and confidence to an actionable signal:     │
│    BUY  — label=UP,   confidence ≥ 65%                      │
│    HOLD — label=UP,   confidence 55–65%                     │
│    SELL — label=DOWN, any confidence                        │
│  Output the signal with top-3 SHAP feature contributors.   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Agentic Properties

| Property | Implementation |
|----------|---------------|
| Autonomy | Executes the full loop from raw data to recommendation without human input |
| Reactivity | Fetches live data on every invocation — no cached or stale inputs |
| Goal-directedness | Produces a single, actionable output optimized for decision clarity |
| Transparency | Prints all 12 computed feature values and top-3 SHAP explanations |
| Robustness | Graceful fallback per stream; system continues if one source is unavailable |

### 6.4 Interfaces

**Command-line interface:**
```bash
python agent/03_agent.py --ticker NVDA
```

**Streamlit web dashboard** (for live class demonstration):
```bash
streamlit run agent/streamlit_app.py
```

The Streamlit dashboard presents three panels: a live price and indicator
chart, a social and macro signal summary, and the BUY/HOLD/SELL recommendation
card with confidence bar and SHAP feature breakdown.

---

## 7. Project Timeline

| Week | Dates | Milestone | Owner |
|------|-------|-----------|-------|
| 1 | Apr 10–13 | All five data streams collected and merged → `master_features.csv` | Person A |
| 2 | Apr 14–19 | Five models trained across two experiments; SHAP computed; `rf_best.pkl` saved | Person B |
| 3 | Apr 21–26 | Agentic CLI and Streamlit dashboard complete; slide deck drafted | Person C |
| 4 | Apr 28 | Full group rehearsal; slide polish; timing confirmed | All |
| 4 | Apr 30 | Final submission on CANVAS; live class presentation | All |

**Critical path:** `master_features.csv` (due Apr 13) is the only inter-member
dependency. Person B and Person C can work in parallel once this file is
available.

---

## 8. Repository Structure

```
stockml-cs513/
├── README.md                     This document
├── requirements.txt
├── data/
│   ├── raw/                      Raw CSVs per ticker
│   └── features/
│       └── master_features.csv   Merged feature matrix (4,800 × 14)
├── src/
│   ├── 01_data_pipeline.py       Data collection and feature engineering
│   └── 02_train_models.py        Model training, evaluation, SHAP, charts
├── agent/
│   ├── 03_agent.py               Four-step agentic CLI loop
│   └── streamlit_app.py          Live three-panel Streamlit dashboard
├── models/
│   └── rf_best.pkl               Serialized model bundle {model, scaler, features}
├── results/
│   ├── results_A.csv             Experiment A: 5 models × 5 metrics
│   ├── results_B.csv             Experiment B: 5 models × 5 metrics
│   └── figures/                  Charts for slides and report
└── slides/
    └── stockml_cs513.pptx        11-slide presentation deck
```

---

## 9. Dependencies

```
yfinance>=0.2.36
ta>=0.11.0
pytrends>=4.9.2
praw>=7.7.1
vaderSentiment>=3.3.2
fredapi>=0.5.1
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
shap>=0.45.0
streamlit>=1.32.0
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.13.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 10. References

1. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002).
   SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial
   Intelligence Research*, 16, 321–357.

2. Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model
   for Sentiment Analysis of Social Media Text. *ICWSM*.

3. Lundberg, S., & Lee, S.I. (2017). A Unified Approach to Interpreting Model
   Predictions. *NeurIPS*.

4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.

5. Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language
   Models. *arXiv:2210.03629*.

6. Federal Reserve Bank of St. Louis. FRED Economic Data API.
   https://fred.stlouisfed.org/docs/api/api_key.html

7. Aroussi, R. yfinance: Yahoo! Finance market data downloader.
   https://github.com/ranaroussi/yfinance

---

> **Disclaimer:** This project is submitted in partial fulfillment of the
> requirements for CS513 at Stevens Institute of Technology. All
> recommendations produced by the agentic system are for educational purposes
> only and do not constitute financial advice.
