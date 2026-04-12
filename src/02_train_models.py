import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ALL_FEATURES,
    DATE_COLUMN,
    FIGURES_DIR,
    MASTER_FEATURES_PATH,
    RESULTS_DIR,
    RF_BEST_MODEL_PATH,
    TARGET_COLUMN,
    TECHNICAL_FEATURES,
    TICKER_COLUMN,
    ensure_directories,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
LOGGER = logging.getLogger("train_models")

MODELS = {
    "kNN": KNeighborsClassifier(n_neighbors=11, weights='distance'),
    "GNB": GaussianNB(),
    "CART": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=500, random_state=42)
}

def build_pipeline(classifier):
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', classifier)
    ])

def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    }

def run_experiment(df, features):
    tscv = TimeSeriesSplit(n_splits=10)
    
    X = df[features]
    y = df[TARGET_COLUMN]
    
    results = []
    
    for name, clf in MODELS.items():
        LOGGER.info("Evaluating %s...", name)
        fold_metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple check for cases where SMOTE might fail due to insufficient targets in early fold
            if y_train.nunique() < 2:
                continue
                
            pipeline = build_pipeline(clf)
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            
            fold_metrics.append(evaluate_model(y_test, y_pred, y_prob))
            
        if fold_metrics:
            avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
            avg_metrics["Model"] = name
            results.append(avg_metrics)
        else:
            LOGGER.warning("Could not gather eval metrics for %s", name)
        
    return pd.DataFrame(results).set_index("Model")

def train_and_explain_best_rf(df, features):
    LOGGER.info("Training final Random Forest (over all data) & creating SHAP Explainability...")
    X = df[features]
    y = df[TARGET_COLUMN]
    
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
    
    # SMOTE over everything prior to finalizing RF model
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    rf.fit(X_res, y_res)
    
    # SHAP Generation
    # Randomly subsample for SHAP to ensure quick, stable plot generation 
    X_sample = X_res.sample(n=min(500, len(X_res)), random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    
    # TreeExplainer behavior depends on SHAP version. Usually returns a list where [1] is positive class
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_class1, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Random Forest - All Streams)")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap_dest = FIGURES_DIR / "shap_summary.png"
    plt.savefig(shap_dest, dpi=300)
    plt.close()
    
    # Save Model Artifacts
    # We explicitly package into a dictionary. This is highly useful for standalone inference logic
    # because predict_time data shouldn't be SMOTEd, meaning using the original pipeline is clumsy.
    bundle = {
        'scaler': scaler,
        'model': rf,
        'features': features
    }
    joblib.dump(bundle, RF_BEST_MODEL_PATH)
    LOGGER.info("Exported Best Model bundle to %s", RF_BEST_MODEL_PATH)
    LOGGER.info("Exported SHAP Summary to %s", shap_dest)

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_directories()
    
    if not MASTER_FEATURES_PATH.exists():
        LOGGER.error("Master features not found at %s. Run Phase 1 first.", MASTER_FEATURES_PATH)
        return
        
    LOGGER.info("Loading master features: %s", MASTER_FEATURES_PATH)
    df = pd.read_csv(MASTER_FEATURES_PATH)
    df.sort_values([DATE_COLUMN, TICKER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    LOGGER.info("=== Starting Experiment A (Technical Features Only) ===")
    results_A = run_experiment(df, TECHNICAL_FEATURES)
    results_A.to_csv(RESULTS_DIR / "results_A.csv")
    LOGGER.info("Experiment A saved.\n%s\n", results_A)
    
    LOGGER.info("=== Starting Experiment B (All Features) ===")
    results_B = run_experiment(df, ALL_FEATURES)
    results_B.to_csv(RESULTS_DIR / "results_B.csv")
    LOGGER.info("Experiment B saved.\n%s\n", results_B)
    
    LOGGER.info("=== Performance Delta ===")
    comparison = pd.DataFrame({
        "Exp_A_Acc": results_A["Accuracy"],
        "Exp_B_Acc": results_B["Accuracy"],
        "Accuracy_Delta": results_B["Accuracy"] - results_A["Accuracy"]
    })
    print(comparison)
    print("\n")
    
    train_and_explain_best_rf(df, ALL_FEATURES)
    LOGGER.info("Phase 2 Target Completion Achieved!")

if __name__ == "__main__":
    main()
