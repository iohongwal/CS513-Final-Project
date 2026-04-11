# Final Results Writeup (CS513)

## Project Summary
This project builds an end-to-end stock direction classification and recommendation workflow:
- Data pipeline combines technical indicators with alternative features.
- Two experiments are evaluated using time-series cross-validation.
- A Random Forest model bundle is exported for live agent inference.
- A CLI and Streamlit dashboard provide recommendation outputs.

## Data And Evaluation Scope
- Tickers: AAPL, TSLA, NVDA, JPM
- Feature table: `data/features/master_features.csv`
- Cross-validation outputs: `results/fold_metrics.csv`

## Experiment Outcome Summary
From `results/results_A.csv` and `results/results_B.csv`:

- Mean accuracy (Experiment A): **0.5221**
- Mean accuracy (Experiment B): **0.5366**
- Accuracy delta (B - A): **+0.0145**

Best accuracy by experiment:
- Experiment A: **GNB** (0.5470)
- Experiment B: **DT** (0.5714)

Best F1 by experiment:
- Experiment A: **MLP** (0.4763)
- Experiment B: **MLP** (0.4224)

## Generated Figures
- Experiment comparison: `results/figures/exp_a_vs_b.png`
- Confusion matrix: `results/figures/confusion_matrix.png`
- SHAP importance: `results/figures/shap.png`

## Demo Evidence
CLI prediction output:
- `submission/cli_demo_output.txt`
- `submission/screenshots/cli_agent_output.png`

Streamlit dashboard screenshot:
- `submission/screenshots/streamlit_dashboard.png`

## Release Tag
Validated baseline tag pushed to remote:
- **v1.0.0**
