# CS513 — Facial Expression Classification with Agentic Component

**Stevens Institute of Technology | Spring 2026**
**Course:** CS513 Data Analytics & Machine Learning

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Folder Structure](#3-folder-structure)
4. [Environment Setup](#4-environment-setup)
5. [Phase 1 — EDA](#5-phase-1--eda)
6. [Phase 2 — Feature Extraction](#6-phase-2--feature-extraction)
7. [Phase 3 — Preprocessing](#7-phase-3--preprocessing)
8. [Phase 4 — Model Training](#8-phase-4--model-training)
9. [Phase 5 — Evaluation](#9-phase-5--evaluation)
10. [Phase 6 — SHAP Explainability](#10-phase-6--shap-explainability)
11. [Phase 7 — Agentic Demo (Streamlit)](#11-phase-7--agentic-demo-streamlit)
12. [Timeline](#12-timeline)
13. [Pitfalls &amp; Tips](#13-pitfalls--tips)

---

## 1. Project Overview

**Problem Statement:**
Classify facial expressions from images into 5 emotion categories using classical
machine learning on extracted visual features, then deploy the best model inside an
emotion-aware agent that generates real-time empathetic responses.

**Research Question:**

> Do richer visual features (HOG + LBP + texture) improve facial expression
> classification accuracy over basic color and brightness features alone?

**Emotions (5 classes):**

| Label | Class    | Approx. Count (FER-2013) |
| ----- | -------- | ------------------------ |
| 0     | angry    | 3,995                    |
| 1     | happy    | 7,215                    |
| 2     | neutral  | 4,965                    |
| 3     | sad      | 4,830                    |
| 4     | surprise | 3,171                    |

> Drop `fear` and `disgust` — too few samples, hurts model stability.

**Dual Experiment Design:**

|                   | Experiment A                      | Experiment B                       |
| ----------------- | --------------------------------- | ---------------------------------- |
| Features          | 6 basic (color + brightness)      | 24 full (color + HOG + LBP + edge) |
| Purpose           | Baseline                          | Full feature set                   |
| Research question | Can color alone classify emotion? | Does texture add signal?           |

---

## 2. Dataset

**Primary:** FER-2013

- Source: https://www.kaggle.com/datasets/msambare/fer2013
- Size: 35,887 grayscale images, 48×48 pixels
- Split: 28,709 train / 7,178 test
- Download: `kaggle datasets download -d msambare/fer2013`

**Alternative (better visual quality):**

- Balanced FER-2013: https://www.kaggle.com/datasets/dollyprajapati182/fer2013-balance-dataset
- RAF-DB (real-world color photos): https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

---

## 3. Folder Structure

```
cs513-facial/
├── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       └── (same structure)
├── features/
│   ├── features_A.csv       ← Experiment A (6 features)
│   └── features_B.csv       ← Experiment B (24 features)
├── models/
│   ├── rf_model.pkl          ← Best model saved
│   └── scaler.pkl            ← Fitted StandardScaler
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation_shap.ipynb
├── app/
│   └── streamlit_app.py      ← Agent demo
└── requirements.txt
```

---

## 4. Environment Setup

### Install dependencies

```bash
python -m venv .venv

# Windows CMD (recommended over PowerShell)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
opencv-python>=4.9.0
scikit-image>=0.22.0
Pillow>=10.3.0
shap>=0.45.0
matplotlib>=3.8.0
seaborn>=0.13.0
streamlit>=1.33.0
ollama>=0.2.0
joblib>=1.4.0
tqdm>=4.66.0
jupyterlab>=4.1.0
```

### Verify installation

```python
import cv2, sklearn, shap
from skimage.feature import hog, local_binary_pattern
print("OpenCV:", cv2.__version__)
print("sklearn:", sklearn.__version__)
print("All OK ✓")
```

---

## 5. Phase 1 — EDA

**Goal:** Understand class distribution and image quality before feature extraction.

### 5.1 Count images per class

```python
# HINT: walk dataset/train/, count files per subfolder
# Expected: happy >> surprise, confirm imbalance
```

### 5.2 Visualize sample images

```python
# HINT: matplotlib subplot grid
# show 5 random images per class → check image quality
```

### 5.3 Plot class distribution bar chart

```python
# HINT: use Counter + plt.bar()
# This justifies using SMOTE in your preprocessing
```

### 5.4 Check pixel value distribution

```python
# HINT: plot histogram of V-channel (brightness)
# across all classes → do happy images look brighter?
```

**Deliverable:** 1 class distribution chart + 1 sample grid image for slides.

---

## 6. Phase 2 — Feature Extraction

**Goal:** Convert each image into a flat feature vector for sklearn.

### 6.1 Feature definitions

| ID      | Feature                    | Code hint                      | Captures           |
| ------- | -------------------------- | ------------------------------ | ------------------ |
| F1      | Brightness mean            | `V.mean()`                   | Overall luminance  |
| F2      | Brightness std             | `V.std()`                    | Lighting variation |
| F3      | Michelson contrast         | `(max-min)/(max+min+eps)`    | Scene clarity      |
| F4      | Saturation mean            | `S.mean()`                   | Color richness     |
| F5      | Dark pixel ratio           | `(V < 50).sum() / size`      | Shadows / sadness  |
| F6      | Hue variance               | `H.var()`                    | Color diversity    |
| —      | *Experiment A ends here* |                                |                    |
| F7      | HOG energy                 | `np.mean(hog_feats**2)`      | Shape + structure  |
| F8–F23 | LBP histogram              | `np.histogram(lbp, bins=16)` | Micro-texture      |
| F24     | Edge density               | `Canny.sum() / size`         | Sharpness / detail |
| —      | *Experiment B ends here* |                                |                    |

### 6.2 Image loading helper

```python
def load_image(path, size=(48, 48)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, gray, hsv
```

### 6.3 Feature extraction function (skeleton)

```python
def extract_features(path):
    img, gray, hsv = load_image(path)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # --- Experiment A (6 basic features) ---
    basic = [
        V.mean(),                          # F1
        V.std(),                           # F2
        # F3: Michelson contrast → your code here
        # F4: saturation mean   → your code here
        # F5: dark pixel ratio  → your code here
        # F6: hue variance      → your code here
    ]

    # --- Experiment B (add 18 more features) ---
    hog_feats = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2))
    hog_energy = ...   # hint: np.mean(hog_feats ** 2)

    lbp = local_binary_pattern(gray, P=8, R=1)
    lbp_hist, _ = np.histogram(lbp, bins=16, range=(0,256), density=True)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = ...   # hint: edges.sum() / edges.size

    full = basic + [hog_energy] + lbp_hist.tolist() + [edge_density]
    return basic, full
```

### 6.4 Build dataset CSV

```python
# HINT: loop structure
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
rows_A, rows_B = [], []

for emotion in EMOTIONS:
    folder = f"data/train/{emotion}/"
    for fname in tqdm(os.listdir(folder), desc=emotion):
        path = folder + fname
        try:
            basic, full = extract_features(path)
            rows_A.append(basic + [emotion])
            rows_B.append(full  + [emotion])
        except Exception as e:
            pass   # skip corrupt files

# Save CSVs
pd.DataFrame(rows_A, columns=[...] + ['label']).to_csv('features/features_A.csv', index=False)
pd.DataFrame(rows_B, columns=[...] + ['label']).to_csv('features/features_B.csv', index=False)
```

> ⚠️ **Always test on ONE image before looping 35K images.**

### 6.5 Validate extracted features

```python
df = pd.read_csv("features/features_B.csv")
print(df.shape)          # expect (~24K rows, 25 cols)
print(df.isnull().sum()) # must be all zeros
print(df.describe())     # sanity check value ranges
print(df["label"].value_counts())  # confirm 5 classes
```

**Expected extraction time:** 20–40 minutes for full FER-2013.
**Save CSV immediately** — do not re-extract.

---

## 7. Phase 3 — Preprocessing

**Goal:** Prepare feature matrix for sklearn pipeline.

### 7.1 Encode labels

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df["label"])
# le.classes_ → ['angry', 'happy', 'neutral', 'sad', 'surprise']
```

### 7.2 Build imblearn Pipeline

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def make_pipeline(clf):
    return Pipeline([
        ("scaler", StandardScaler()),   # Z-score normalize
        ("smote",  SMOTE(random_state=42)),   # fix class imbalance
        ("clf",    clf)
    ])
```

> ✅ SMOTE inside Pipeline ensures it only applies to training folds, not test folds.

### 7.3 Cross-validation setup

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

---

## 8. Phase 4 — Model Training

**Goal:** Train 5 classifiers on both experiments, collect all metrics.

### 8.1 Model definitions

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

MODELS = {
    "kNN":  KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "GNB":  GaussianNB(),
    "CART": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RF":   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "MLP":  MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42),
}
```

### 8.2 Scoring metrics

```python
SCORING = {
    "accuracy":  "accuracy",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
    "f1":        "f1_weighted",
    "roc_auc":   "roc_auc_ovr_weighted",
}
```

### 8.3 Training loop (hint)

```python
from sklearn.model_selection import cross_validate

results = {}
for exp, csv_path in [("A","features/features_A.csv"),
                      ("B","features/features_B.csv")]:
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values
    y = le.fit_transform(df["label"])

    for name, clf in MODELS.items():
        pipe   = make_pipeline(clf)
        scores = cross_validate(pipe, X, y, cv=cv,
                                scoring=SCORING, n_jobs=-1)
        results[f"{exp}_{name}"] = {
            m: round(scores[f"test_{m}"].mean(), 4) for m in SCORING
        }
        print(f"[Exp {exp}] {name} Acc={results[f'{exp}_{name}']['accuracy']:.3f}")

pd.DataFrame(results).T.to_csv("results_summary.csv")
```

### 8.4 Training order (by speed)

```
1. GNB      → ~2 sec   ← run first to confirm pipeline works
2. CART     → ~1 min
3. kNN      → ~3 min
4. RF       → ~10 min
5. MLP      → ~20 min  ← run last
```

### 8.5 Save best model

```python
import joblib
# After training, refit RF on full dataset (not just folds)
# HINT: scaler.fit_transform(X) → rf.fit(X_scaled, y)
joblib.dump(rf_final, "models/rf_model.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
```

---

## 9. Phase 5 — Evaluation

**Goal:** Compare 5 models × 2 experiments, produce presentation visuals.

### 9.1 Results table

```python
df_results = pd.read_csv("results_summary.csv", index_col=0)
print(df_results[["accuracy","f1","roc_auc"]].to_markdown())
```

### 9.2 Confusion matrix (RF best model)

```python
# HINT: use cross_val_predict to get predictions across all folds
# then ConfusionMatrixDisplay.from_predictions()
from sklearn.metrics import ConfusionMatrixDisplay
```

### 9.3 Per-fold accuracy plot

```python
# HINT: plot all 10 fold accuracy scores as line chart
# one line per model → shows stability
```

### 9.4 Experiment A vs B comparison bar chart

```python
# HINT: grouped bar chart
# x-axis = model names, bars = Exp A vs Exp B accuracy
# this is your key research finding slide
```

---

## 10. Phase 6 — SHAP Explainability

**Goal:** Explain which visual features drive each emotion prediction.

### 10.1 SHAP setup (RF only)

```python
import shap
explainer   = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_scaled[:500])   # sample 500 rows
```

### 10.2 Summary plot (global feature importance)

```python
# HINT: shap.summary_plot(shap_values, X_sample,
#                         feature_names=feature_names,
#                         class_names=le.classes_)
```

### 10.3 Expected SHAP story per emotion

| Emotion     | Top driving features                              |
| ----------- | ------------------------------------------------- |
| 😊 Happy    | High edge density, high brightness, high LBP f3   |
| 😢 Sad      | High dark ratio, low brightness std, low contrast |
| 😠 Angry    | High HOG energy, high LBP f8–f10, high contrast  |
| 😲 Surprise | Very high HOG energy, very high edge density      |
| 😐 Neutral  | All mid-range features, low LBP variance          |

---

## 11. Phase 7 — Agentic Demo (Streamlit)

**Goal:** Upload face photo → classify emotion → agent response.

### 11.1 Agent loop

```
OBSERVE  → user uploads face image via Streamlit
THINK    → extract 24 features using same pipeline
ACT      → RF predicts emotion + confidence %
EXPLAIN  → SHAP top-3 features + Gemma LLM response
```

### 11.2 Streamlit app skeleton

```python
import streamlit as st
import joblib, cv2, numpy as np

rf    = joblib.load("models/rf_model.pkl")
scaler= joblib.load("models/scaler.pkl")
LABELS= ['angry','happy','neutral','sad','surprise']

st.title("🎭 Facial Expression Classifier")
uploaded = st.file_uploader("Upload a face photo", type=["jpg","png"])

if uploaded:
    # HINT: decode image bytes with cv2.imdecode
    # extract features, scale, predict, show results
    # show: emotion label, confidence bar, top SHAP features
    pass
```

### 11.3 Agent responses (fallback without Ollama)

```python
RESPONSES = {
    "happy":    "😊 You seem happy! Great energy.",
    "sad":      "😢 You seem sad. Things will improve.",
    "angry":    "😠 You seem frustrated. Take a short break.",
    "surprise": "😲 Something unexpected happened?",
    "neutral":  "😐 Neutral detected. All systems normal.",
}
```

### 11.4 Run demo

```bash
streamlit run app/streamlit_app.py
```

---

## 12. Timeline

| Date       | Phase              | Task                                                                  | Owner |
| ---------- | ------------------ | --------------------------------------------------------------------- | ----- |
| Apr 24     | EDA                | Download FER-2013, class distribution chart, sample grid              | A     |
| Apr 25     | Feature Extraction | `extract_features()`, build `features_A.csv` + `features_B.csv` | A     |
| Apr 25–26 | Training           | Train 5 models × 2 experiments, save results CSV                     | B     |
| Apr 26     | Evaluation         | Confusion matrix, A vs B bar chart, per-fold plot                     | B     |
| Apr 26–27 | SHAP               | SHAP summary plot + per-class analysis                                | B     |
| Apr 27–28 | Agent Demo         | Streamlit app + response layer                                        | C     |
| Apr 29     | Slides             | Full dry run, finalize presentation deck                              | All   |
| Apr 30     | Present            | Submit + present                                                      | All   |

---

## 13. Pitfalls & Tips

### ❌ Common Mistakes

- Running `StandardScaler` before `train_test_split` → data leakage
- Applying SMOTE to test fold → use `imblearn.Pipeline`, not `sklearn.Pipeline`
- Skipping single-image test before full loop → wastes 30+ minutes if broken
- Using `roc_auc` without `ovr_weighted` for multi-class → will crash
- Forgetting `n_jobs=-1` on RF and kNN → 4–8× slower

### ✅ Best Practices

- Test pipeline on 100 images first → confirm shape and no NaN
- Save `features_B.csv` immediately after extraction → do not re-run
- Use `tqdm` progress bar in extraction loop → monitor progress
- Print `.std()` of fold scores alongside `.mean()` → shows stability
- Save model with `joblib.dump()` → needed for Streamlit demo

### 🔑 Key Imports Cheatsheet

```python
import os, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import shap, joblib
```

---
