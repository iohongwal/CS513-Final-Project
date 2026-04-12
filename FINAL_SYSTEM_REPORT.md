# CS513 Final Project: StockML Autonomous Agent
## Final System Architecture & Deployment Guide

This document serves as the comprehensive final report detailing the architecture, configuration, and deployment procedures for the **StockML Autonomous Agent**, developed as the final project for CS513 Data Mining.

---

## 1. System Overview

The StockML Assistant is an end-to-end Machine Learning pipeline and autonomous ReAct agent designed to predict daily equity movements (UP/DOWN) for tech-heavy tickers (`AAPL`, `TSLA`, `NVDA`) and traditional banking (`JPM`).

Rather than relying purely on classical technical indicators (moving averages, RSI), this system proves that **Data Mining Alternative Signals** across the internet improves classification accuracy.

### The Three Phases:
1. **Phase 1: Data Engineering** (`01_data_pipeline.py`) - Extracts and unifies Google Trends, Reddit Sentiment, FRED Macroeconomic Rates, and Yahoo Finance prices.
2. **Phase 2: Mathematical Modeling** (`02_train_models.py`) - Evaluates kNN, CART, Naive Bayes, MLP, and Random Forest using chronological `TimeSeriesSplit`. It bundles the best performing model.
3. **Phase 3: The Deployed Agent** (`03_agent.py` & `streamlit_app.py`) - A live interface that dynamically streams today's data, normalizes it using a pre-saved `MinMaxScaler`, and calculates the most prominent driving factors using **SHAP TreeExplainer**.

> **Note:** To strictly adhere to the classical Data Mining curriculum, the deployed Agent leverages complex Random Forest classification and programmatic logic to infer sentiment rather than relying on generative AI/LLMs.

---

## 2. Configuration & Setup

### Environment Requirements
Ensure Python 3.10+ is installed on your machine.
1. Clone this repository locally.
2. Activate a virtual environment.
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### API Key Management
Because the system streams live internet data, it requires programmatic access to external endpoints. You must configure a `.env` file in the root directory.

Create a file exactly named `.env` and populate it using the provided `.env.example` file template:
```env
# Reddit Setup
REDDIT_CLIENT_ID="your_client_id_here"
REDDIT_CLIENT_SECRET="your_secret_here"
REDDIT_USER_AGENT="cs513-scraper"

# FRED Interface Setup
FRED_API_KEY="your_api_key_here"
```
*(If any keys are missing, the system's dynamic error-handling will automatically replace missing data sets with zero-padded medians to prevent catastrophic pipeline failure).*

---

## 3. Execution Pipeline

The repository must be executed in chronological order for the dependent files to be generated properly.

### Step 1: Re-build the Historical Base
Run the data pipeline to scrape the last 5 years of alternative data across the internet. This will generate `master_features.csv`.
```bash
python src/01_data_pipeline.py
```

### Step 2: Retrain the AI Models
Push the CSV dataset through our 5 machine learning models. This will execute `SMOTE` oversampling natively, output `results_A.csv` and `results_B.csv`, generate global `SHAP` interpretation plots, and mathematical encapsulate the Random Forest into `rf_best.pkl`.
```bash
python src/02_train_models.py
```

### Step 3: Launch the Agentic System
The autonomous agent is designed to execute a 4-step logic cycle (**Observe** -> **Think** -> **Act** -> **Recommend**) out of the box.

**Command-Line Interface (CLI):**
To execute a headless analysis of a stock natively in the terminal:
```bash
python agent/03_agent.py --ticker TSLA
```
*Expected Output: A BUY/SELL signal alongside a dynamically calculated Confidence metrics and the TOP 3 factors driving that exact decision.*

**Interactive Streamlit Dashboard:**
To boot up the presentation-ready visual interface:
```bash
streamlit run agent/streamlit_app.py
```
*(Note: Because the backend must dynamically ping Reddit, Google Trends, and FRED live via network requests upon clicking 'Analyze', the dashboard includes a ~15 second loading state spinner).*

---

## 4. Deployed Technologies

* **Data Engineering**: `praw`, `pytrends`, `yfinance`, `vaderSentiment`, `fredapi`
* **Modeling / Classification**: `scikit-learn` (RandomForest, TimeSeriesSplit), `imblearn` (SMOTE)
* **Interpretability**: `shap` (TreeExplainer)
* **Frontend**: `streamlit`, `matplotlib`
