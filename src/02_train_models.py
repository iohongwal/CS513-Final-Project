from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
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

LOGGER = logging.getLogger("train_models")


@dataclass
class ExperimentResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    tuned_params: dict[str, dict[str, Any]]


def load_master_features() -> pd.DataFrame:
    if not MASTER_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing input feature table: {MASTER_FEATURES_PATH}")

    df = pd.read_csv(MASTER_FEATURES_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df.sort_values([DATE_COLUMN, TICKER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_models(tuned_params: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    tuned_params = tuned_params or {}

    dt_params = {"max_depth": 6, "criterion": "gini", "min_samples_leaf": 20}
    dt_params.update(tuned_params.get("dt", {}))

    rf_params = {"n_estimators": 100, "max_features": "sqrt"}
    rf_params.update(tuned_params.get("rf", {}))

    mlp_params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "max_iter": 500,
        "early_stopping": True,
    }
    mlp_params.update(tuned_params.get("mlp", {}))

    return {
        "knn": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(random_state=42, **dt_params),
        "rf": RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params),
        "mlp": MLPClassifier(random_state=42, **mlp_params),
    }


def _resolve_tscv_splits(n_samples: int, preferred: int) -> int:
    max_allowed = n_samples - 1
    if max_allowed < 2:
        raise ValueError("Need at least 3 rows for TimeSeriesSplit")
    return min(preferred, max_allowed)


def _select_resampler(y: pd.Series) -> RandomOverSampler | SMOTE | None:
    counts = y.value_counts()
    if len(counts) < 2:
        return None

    minority_count = int(counts.min())
    if minority_count < 2:
        return RandomOverSampler(random_state=42)

    return SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))


def _build_training_pipeline(model: Any, y_train: pd.Series) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", MinMaxScaler())]

    sampler = _select_resampler(y_train)
    if sampler is not None:
        steps.append(("resample", sampler))

    steps.append(("model", model))
    return Pipeline(steps)


def tune_key_models(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, Any]]:
    LOGGER.info("Light hyperparameter tuning started for dt/rf/mlp")

    cv = TimeSeriesSplit(n_splits=_resolve_tscv_splits(len(X), preferred=3))
    tuned: dict[str, dict[str, Any]] = {}

    searches = {
        "dt": (
            DecisionTreeClassifier(random_state=42),
            {
                "model__max_depth": [4, 6, 8],
                "model__min_samples_leaf": [10, 20, 30],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8, 12],
                "model__max_features": ["sqrt", 0.7],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "mlp": (
            MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
            {
                "model__hidden_layer_sizes": [(64, 32), (128, 64)],
                "model__alpha": [0.001, 0.0005],
                "model__learning_rate_init": [0.001, 0.0005],
            },
        ),
    }

    for model_name, (estimator, grid) in searches.items():
        pipe = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                # RandomOverSampler is more stable for very small CV folds during tuning.
                ("resample", RandomOverSampler(random_state=42)),
                ("model", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X, y)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items() if k.startswith("model__")}
        tuned[model_name] = best
        LOGGER.info("Best %s params: %s", model_name, best)

    return tuned


def _score(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return result


def evaluate_experiment(df: pd.DataFrame, feature_cols: list[str], experiment: str) -> ExperimentResult:
    X = df[feature_cols].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    tuned = tune_key_models(X, y)
    models = build_models(tuned)

    cv = TimeSeriesSplit(n_splits=_resolve_tscv_splits(len(X), preferred=10))
    rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if y_train.nunique() < 2:
            LOGGER.warning("Skipping fold %d in experiment %s due single-class training window", fold, experiment)
            continue

        for name, base_model in models.items():
            model = clone(base_model)
            if name == "knn":
                max_neighbors = max(1, len(X_train))
                model.set_params(n_neighbors=min(model.n_neighbors, max_neighbors))

            pipe = _build_training_pipeline(model, y_train)
            try:
                pipe.fit(X_train, y_train)
            except ValueError as exc:
                LOGGER.warning("Skipping %s on fold %d due training error: %s", name, fold, exc)
                continue

            y_pred = pipe.predict(X_test)
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(X_test)
                if probs.shape[1] == 1:
                    model_classes = pipe.named_steps["model"].classes_
                    y_prob = np.ones(len(X_test)) if int(model_classes[0]) == 1 else np.zeros(len(X_test))
                else:
                    y_prob = probs[:, 1]
            else:
                raw = pipe.decision_function(X_test)
                y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

            metrics = _score(y_test.to_numpy(), y_pred, y_prob)
            rows.append({"experiment": experiment, "model": name, "fold": fold, **metrics})

    fold_df = pd.DataFrame(rows)
    summary_df = (
        fold_df.groupby(["experiment", "model"], as_index=False)[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        .mean()
        .sort_values(["experiment", "f1"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return ExperimentResult(fold_metrics=fold_df, summary=summary_df, tuned_params=tuned)


def plot_experiment_comparison(summary_a: pd.DataFrame, summary_b: pd.DataFrame) -> None:
    merged = pd.concat([summary_a, summary_b], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sns.barplot(data=merged, x="model", y="accuracy", hue="experiment", ax=axes[0])
    axes[0].set_title("Accuracy")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=merged, x="model", y="f1", hue="experiment", ax=axes[1])
    axes[1].set_title("F1")
    axes[1].tick_params(axis="x", rotation=25)

    sns.barplot(data=merged, x="model", y="roc_auc", hue="experiment", ax=axes[2])
    axes[2].set_title("ROC-AUC")
    axes[2].tick_params(axis="x", rotation=25)

    fig.savefig(FIGURES_DIR / "exp_a_vs_b.png", dpi=220)
    plt.close(fig)


def train_rf_bundle(df: pd.DataFrame, feature_cols: list[str], tuned_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    X = df[feature_cols].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    sampler = _select_resampler(y)
    if sampler is None:
        X_bal, y_bal = X_scaled, y.to_numpy()
    else:
        X_bal, y_bal = sampler.fit_resample(X_scaled, y)

    params = {"n_estimators": 100, "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    params.update(tuned_params.get("rf", {}))

    rf = RandomForestClassifier(**params)
    rf.fit(X_bal, y_bal)

    bundle = {
        "model": rf,
        "scaler": scaler,
        "features": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "random_forest",
        "params": params,
    }
    RF_BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, RF_BEST_MODEL_PATH)
    return bundle


def plot_confusion_matrix(df: pd.DataFrame, feature_cols: list[str], tuned_params: dict[str, dict[str, Any]]) -> None:
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[feature_cols].astype(float)
    y_train = train[TARGET_COLUMN].astype(int)
    X_test = test[feature_cols].astype(float)
    y_test = test[TARGET_COLUMN].astype(int)

    if y_train.nunique() < 2 or y_test.empty:
        LOGGER.warning("Skipping confusion matrix due insufficient class variation in holdout split")
        return

    model_params = {"n_estimators": 100, "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    model_params.update(tuned_params.get("rf", {}))

    pipe = _build_training_pipeline(RandomForestClassifier(**model_params), y_train)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    matrix = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (RF, Experiment B Holdout)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shap_importance(bundle: dict[str, Any], df: pd.DataFrame) -> None:
    try:
        import shap
    except Exception:
        LOGGER.warning("shap not available; skipping SHAP plot")
        return

    X = df[bundle["features"]].astype(float)
    X_scaled = bundle["scaler"].transform(X)

    sample_size = min(800, len(X_scaled))
    sample_idx = np.linspace(0, len(X_scaled) - 1, sample_size, dtype=int)
    sample = X_scaled[sample_idx]

    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], sample, feature_names=bundle["features"], plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, sample, feature_names=bundle["features"], plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap.png", dpi=220)
    plt.close()


def run_training() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_directories()

    df = load_master_features()

    exp_a = evaluate_experiment(df, TECHNICAL_FEATURES, "A")
    exp_b = evaluate_experiment(df, ALL_FEATURES, "B")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_a.summary.to_csv(RESULTS_DIR / "results_A.csv", index=False)
    exp_b.summary.to_csv(RESULTS_DIR / "results_B.csv", index=False)

    combined_folds = pd.concat([exp_a.fold_metrics, exp_b.fold_metrics], ignore_index=True)
    combined_folds.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)

    plot_experiment_comparison(exp_a.summary, exp_b.summary)
    plot_confusion_matrix(df, ALL_FEATURES, exp_b.tuned_params)

    bundle = train_rf_bundle(df, ALL_FEATURES, exp_b.tuned_params)
    plot_shap_importance(bundle, df)

    avg_a = exp_a.summary["accuracy"].mean()
    avg_b = exp_b.summary["accuracy"].mean()
    LOGGER.info("Mean accuracy A: %.4f", avg_a)
    LOGGER.info("Mean accuracy B: %.4f", avg_b)
    LOGGER.info("Delta (B-A): %.4f", avg_b - avg_a)
    LOGGER.info("Saved model bundle: %s", RF_BEST_MODEL_PATH)


if __name__ == "__main__":
    run_training()
