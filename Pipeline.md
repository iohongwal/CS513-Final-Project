# CS513 — Facial Expression Classification

## Implementation Pipeline v6.0 (LBP Dropped)

### Spring 2026 | FER-2013 | 5-Class | 18 Features

---

## Feature Set (Final — 18 Features)

| ID  | Name               | Family        | EDA Signal   | Keep?            |
| --- | ------------------ | ------------- | ------------ | ---------------- |
| F01 | brightness_mean    | Intensity     | 1.20x Mod    | ✅               |
| F02 | lower_edge_density | Edge          | 1.11x Mod    | ✅               |
| F03 | hog_h00            | HOG histogram | 1.04x Weak   | ✅ (interaction) |
| F04 | hog_h01            | HOG histogram | 1.04x Weak   | ✅ (interaction) |
| F05 | hog_h02            | HOG histogram | 1.04x Weak   | ✅ (interaction) |
| F06 | hog_h03            | HOG histogram | 1.05x Weak   | ✅ (interaction) |
| F07 | hog_h04            | HOG histogram | 1.01x Weak   | ✅ (interaction) |
| F08 | hog_h05            | HOG histogram | 1.05x Weak   | ✅ (interaction) |
| F09 | hog_h06            | HOG histogram | 1.05x Weak   | ✅ (interaction) |
| F10 | hog_h07            | HOG histogram | 1.16x Mod    | ✅               |
| F11 | hog_h08            | HOG histogram | 1.06x Weak   | ✅ (interaction) |
| F12 | hog_h09            | HOG histogram | 1.08x Weak   | ✅ (interaction) |
| F13 | hog_h10            | HOG histogram | 1.10x Mod    | ✅               |
| F14 | hog_h11            | HOG histogram | 1.14x Mod    | ✅               |
| F15 | hog_h12            | HOG histogram | 1.29x Mod    | ✅               |
| F16 | hog_h13            | HOG histogram | 1.53x Strong | ✅               |
| F17 | hog_h14            | HOG histogram | 1.50x Strong | ✅               |
| F18 | hog_h15            | HOG histogram | 1.81x Strong | ✅               |

**Dropped:**

- ❌ saturation_mean        → 0.00x (grayscale = always 0)
- ❌ upper_edge_density     → 1.05x (no region-level separation)

---

## Phase 1 — EDA (Completed ✅)

**Goal:** Validate that features carry class-separating signal before building anything.

### Step 1 — Folder structure

```python
import os
TRAIN_DIR = "data/train/"
TEST_DIR  = "data/test/"
EMOTIONS  = ["angry", "happy", "neutral", "sad", "surprise"]
print({e: len(os.listdir(TRAIN_DIR + e)) for e in EMOTIONS})
```

### Step 2 — Class counts

```
angry:    3,995   (smallest)
happy:    7,215   (largest)
neutral:  4,965
sad:      4,830
surprise: 3,171

Imbalance ratio: 7215 / 3171 = 2.27x  → must use SMOTE in Phase 3
```

### Step 3 — Class distribution plots

```python
# Bar chart + pie chart
# → saved: eda_01_class_distribution.png
```

### Step 4 — Sample image grid

```python
# 5 rows × 5 cols, one row per emotion
# → saved: eda_02_sample_grid.png
```

### Step 5 — Image dimensions

```
All images: 48×48 pixels, grayscale → no resizing needed
No corrupt files found
```

### Step 6A — Brightness mean

```
F01 brightness_mean: surprise=max, sad=min, ratio=1.20x → KEEP
```

### Step 6B — Saturation mean

```
F_sat saturation_mean: ratio=0.00x → DROP (grayscale image, always 0)
```

### Step 6C — HOG single energy

```
HOG energy collapsed to 1 value: ratio=1.00x → REDESIGN as 16-bin histogram
```

### Step 6D — Edge density (regional)

```
upper_edge_density: surprise=max, neutral=min, ratio=1.05x → DROP
lower_edge_density: happy=max,   neutral=min, ratio=1.11x → KEEP (F02)
```

### Step 6E — HOG histogram (16 bins)

```
Bins 13–15: 1.50x–1.81x ✅ Strong   → high-gradient bins separate angry/sad vs surprise
Bins 7,10–12: 1.10x–1.29x ⚠️  Mod
Bins 0–6, 8–9: ~1.01–1.08x ❌ Weak  → kept for interaction effects

Key finding: HOG signal is right-skewed — emotion separability lives in bins 11–15
(strong-gradient regime = facial muscle tension from angry/sad)
```

### Step 6F — LBP histogram (16 bins) — DROPPED

```
After fixing uniform→default bug:
  Max ratio across all 16 bins: 1.18x (lbp_h05 and lbp_h11)
  LBP is suppressed by HOG which already captures texture orientation
  Decision: DROP all 16 LBP bins from feature set
  Final feature count: 18 (down from 34)
```

### Step 6G — Full feature signal report

```
✅ Strong  (>1.3x):  3   →  hog_h13, hog_h14, hog_h15
⚠️  Moderate (1.1–1.3x): 6   →  hog_h07, hog_h10, hog_h11, hog_h12, brightness_mean, lower_edge
❌ Weak   (<1.1x):   9   →  hog_h00–06, hog_h08–09  (kept for RF interaction)

Total features: 18
```

### Step 7 — Feature boxplots

```python
# 4-panel boxplot: brightness, lower_edge, hog_h13, hog_h15
# → saved: eda_03_feature_families.png  (already generated)
```

### EDA Checklist

```
✅ Step 1   Folder structure verified
✅ Step 2   Class counts + 2.27x imbalance documented
✅ Step 3   Bar + pie chart  → eda_01_class_distribution.png
✅ Step 4   Sample grid      → eda_02_sample_grid.png
✅ Step 5   All 48×48 grayscale, no corrupt files
✅ Step 6A  Brightness 1.20x → KEEP F01
✅ Step 6B  Saturation 0.00x → DROP
✅ Step 6C  HOG energy 1.00x → REDESIGN to histogram
✅ Step 6D  Edge density: upper DROP, lower KEEP F02
✅ Step 6E  HOG 16 bins → eda_04_hog_histogram.png + eda_05_hog_bin_ratios.png
✅ Step 6F  LBP 16 bins → MAX 1.18x → DROP all LBP
✅ Step 6G  Full 18-feature signal report printed
✅ Step 7   Boxplot → eda_03_feature_families.png
✅ Step 8   Feature decision finalized: 18 features
```

---

## Phase 2 — Feature Extraction

**Goal:** Extract 18 features per image, save to features.csv.

### Constants

```python
import cv2, numpy as np, os, csv
from skimage.feature import hog

TRAIN_DIR  = "data/train/"
TEST_DIR   = "data/test/"
EMOTIONS   = ["angry", "happy", "neutral", "sad", "surprise"]
IMG_SIZE   = (48, 48)
N_HOG_BINS = 16
HOG_RANGE  = (0, 0.5)

FEATURE_NAMES = (
    ["brightness_mean", "lower_edge_density"] +
    [f"hog_h{i:02d}" for i in range(16)]
)  # len = 18
```

### Core function: extract_features()

```python
def extract_features(img_path: str) -> np.ndarray:
    """
    Input : path to any grayscale or BGR face image
    Output: 1D numpy array of shape (18,)

    Features:
      [0]     brightness_mean     — mean pixel intensity (0–255)
      [1]     lower_edge_density  — Canny edge density in bottom 50% rows
      [2:18]  hog_h00–hog_h15     — 16-bin HOG magnitude histogram
    """
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    # F01 — Brightness
    brightness = gray.mean()

    # F02 — Lower edge density
    edges      = cv2.Canny(gray, threshold1=50, threshold2=150)
    h          = gray.shape[0]
    lower_half = edges[h // 2:, :]
    lower_edge = lower_half.mean() / 255.0

    # F03–F18 — HOG histogram (16 bins)
    hog_desc, _ = hog(gray,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      visualize=True,
                      feature_vector=True)
    hog_hist, _ = np.histogram(hog_desc,
                                bins=N_HOG_BINS,
                                range=HOG_RANGE,
                                density=True)

    return np.array([brightness, lower_edge] + hog_hist.tolist())
```

### Build features.csv (train + test)

```python
def build_csv(split: str, out_path: str):
    """
    split   : "train" or "test"
    out_path: e.g. "features_train.csv"
    """
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    rows = []

    for emotion in EMOTIONS:
        folder = base_dir + emotion + "/"
        for fname in os.listdir(folder):
            fpath = folder + fname
            feats = extract_features(fpath)
            rows.append(list(feats) + [emotion])

    header = FEATURE_NAMES + ["label"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {out_path}")

build_csv("train", "features_train.csv")
build_csv("test",  "features_test.csv")
```

### Expected output

```
features_train.csv  — ~28,821 rows × 19 cols (18 features + label)
features_test.csv   —  ~7,178 rows × 19 cols
Runtime             — ~5–10 min on CPU (no GPU needed)
```

### Validation checks (run after build)

```python
import pandas as pd
df = pd.read_csv("features_train.csv")

print(df.shape)              # should be (28821, 19)
print(df.isnull().sum())     # should be 0 everywhere
print(df["label"].value_counts())
print(df[FEATURE_NAMES].describe())
# hog_h00 mean should be highest (most descriptors are near 0)
# hog_h15 mean should be lowest  (few strong gradients)
```

---

## Phase 3 — Preprocessing

**Goal:** Encode labels, scale features, balance classes with SMOTE.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold

# Load
df_train = pd.read_csv("features_train.csv")
df_test  = pd.read_csv("features_test.csv")

FEATURE_NAMES = [c for c in df_train.columns if c != "label"]

X_train_raw = df_train[FEATURE_NAMES].values
X_test_raw  = df_test[FEATURE_NAMES].values

# Label encode
le = LabelEncoder()
y_train = le.fit_transform(df_train["label"])
y_test  = le.transform(df_test["label"])
# Classes: angry=0, happy=1, neutral=2, sad=3, surprise=4

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Preprocessing pipeline (used inside each CV fold)
preproc = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42, k_neighbors=5))
])
# Note: fit preproc on TRAIN fold only, transform both train and val folds
# Apply scaler (no SMOTE) to test set at final evaluation
```

### Why SMOTE here

```
Before SMOTE → angry: 3,995 | surprise: 3,171 (minority)
After SMOTE  → all classes balanced to ~7,215 samples each
SMOTE operates in feature space (18-dim) — safe for HOG histogram features
```

---

## Phase 4 — Model Training

**Goal:** Train 3 models with 5-fold CV. Capture fold-level metrics.

### Models

| Model         | Key hyperparameters                                     | Why it fits this problem                             |
| ------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| Random Forest | n_estimators=300, max_depth=None, class_weight=balanced | Handles feature interactions; gives SHAP importances |
| MLP           | hidden=(256,128,64), activation=relu, max_iter=500      | Captures non-linear combinations of HOG bins         |
| SVM           | kernel=rbf, C=10, gamma=scale, class_weight=balanced    | Strong baseline on low-dim (18-feat) spaces          |

### Training loop

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np, pandas as pd

models = {
    "RF":  RandomForestClassifier(n_estimators=300, random_state=42,
                                   class_weight="balanced", n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                          activation="relu", max_iter=500, random_state=42),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale",
               class_weight="balanced", random_state=42)
}

results = []
fold_metrics = []

for model_name, clf in models.items():
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train)):
        X_tr, y_tr   = X_train_raw[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train_raw[val_idx], y_train[val_idx]

        # Scale
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # SMOTE (train fold only)
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        # Train
        clf.fit(X_tr, y_tr)

        # Evaluate on val fold
        y_pred = clf.predict(X_val)
        acc    = accuracy_score(y_val, y_pred)
        f1     = f1_score(y_val, y_pred, average="weighted")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_metrics.append({"model": model_name, "fold": fold+1,
                              "acc": acc, "f1": f1})
        print(f"  {model_name} fold {fold+1}: acc={acc:.4f} f1={f1:.4f}")

    results.append({
        "model": model_name,
        "cv_acc_mean": np.mean(fold_accs),
        "cv_acc_std":  np.std(fold_accs),
        "cv_f1_mean":  np.mean(fold_f1s),
        "cv_f1_std":   np.std(fold_f1s)
    })

# Save
pd.DataFrame(results).to_csv("results.csv", index=False)
pd.DataFrame(fold_metrics).to_csv("fold_metrics.csv", index=False)
print("\nResults saved.")
```

### Expected CV performance (baseline targets)

```
RF:   acc ~0.48–0.54,  f1 ~0.46–0.52
MLP:  acc ~0.50–0.56,  f1 ~0.48–0.54
SVM:  acc ~0.46–0.52,  f1 ~0.44–0.50

Note: FER-2013 is a hard dataset even for CNNs (~72% with deep models)
      Classical ML on 18 handcrafted features → 50% is good, 55% is excellent
```

---

## Phase 5 — Evaluation

**Goal:** Final test-set evaluation of best CV model. Generate confusion matrix.

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# Retrain best model on full training set
best_model_name = "MLP"   # replace with actual best from results.csv
clf = models[best_model_name]

# Final scaler + SMOTE on full train set
scaler_final = StandardScaler()
X_tr_scaled  = scaler_final.fit_transform(X_train_raw)
sm_final     = SMOTE(random_state=42, k_neighbors=5)
X_tr_bal, y_tr_bal = sm_final.fit_resample(X_tr_scaled, y_train)
clf.fit(X_tr_bal, y_tr_bal)

# Evaluate on test set (scale only, no SMOTE)
X_te_scaled = scaler_final.transform(X_test_raw)
y_pred_test = clf.predict(X_te_scaled)

# Report
print(classification_report(y_test, y_pred_test,
                             target_names=le.classes_))

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_model_name} (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
```

### Results table (template)

```
Model   CV Acc ± Std    CV F1 ± Std    Test Acc    Test F1
RF      ??.?? ± 0.0?    ??.?? ± 0.0?   ??.??       ??.??
MLP     ??.?? ± 0.0?    ??.?? ± 0.0?   ??.??       ??.??
SVM     ??.?? ± 0.0?    ??.?? ± 0.0?   ??.??       ??.??
```

### Per-class analysis (what to look for)

```
Expected confusion patterns:
  angry   ↔ sad       high confusion (similar low-brightness, high-gradient)
  neutral ↔ sad       moderate confusion (low expressivity)
  surprise → happy    some confusion (both bright, open-face expressions)
  happy               typically highest precision (strong Duchenne smile signal)
```

---

## Phase 6 — SHAP Analysis

**Goal:** Explain which of the 18 features drive each emotion prediction.

```python
import shap, numpy as np, matplotlib.pyplot as plt

# Use RF — SHAP TreeExplainer is exact and fast for RF
rf_clf = models["RF"]  # must be fitted on full train set first

explainer  = shap.TreeExplainer(rf_clf)
shap_vals  = explainer.shap_values(X_te_scaled[:500])
# shap_vals: list of 5 arrays, shape (500, 18) — one per class
```

### Plot 1 — Global feature importance (all classes)

```python
shap.summary_plot(shap_vals, X_te_scaled[:500],
                  feature_names=FEATURE_NAMES,
                  class_names=le.classes_,
                  plot_type="bar")
plt.savefig("shap_summary_bar.png", dpi=150, bbox_inches="tight")
```

**Expected SHAP ranking:**

```
1. hog_h15        ← 1.81x EDA signal → should be #1 SHAP
2. hog_h13        ← 1.53x
3. hog_h14        ← 1.50x
4. hog_h12        ← 1.29x
5. brightness_mean ← 1.20x
6. hog_h11        ← 1.14x
7. lower_edge_density ← 1.11x
...
18. hog_h04       ← 1.01x (near-zero SHAP expected)
```

### Plot 2 — Per-emotion beeswarm (one per class)

```python
class_names = le.classes_
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for i, emotion in enumerate(class_names):
    plt.sca(axes[i])
    shap.summary_plot(shap_vals[i], X_te_scaled[:500],
                      feature_names=FEATURE_NAMES,
                      show=False, max_display=10)
    axes[i].set_title(emotion)

plt.tight_layout()
plt.savefig("shap_per_emotion.png", dpi=150, bbox_inches="tight")
```

### Story to write in report (per emotion)

```
angry:    hog_h15 ↑, hog_h13 ↑ → strong facial muscle tension, deep creases
happy:    brightness_mean ↑, hog_h15 ↓ → bright raised cheeks, smooth skin
neutral:  all features near 0 SHAP → balanced, no dominant pattern
sad:      hog_h15 ↑, brightness ↓ → sharp edges + dark face (brow furrow)
surprise: brightness_mean ↑, lower_edge ↑ → bright wide-open eyes + jaw drop
```

---

## Phase 7 — Agent Demo (Streamlit)

**Goal:** Interactive demo where user uploads a face image → agent predicts emotion + explains it.

### File structure

```
cs513-facial/
├── agent_app.py          ← main Streamlit app
├── agent_core.py         ← prediction + explanation logic
├── models/
│   ├── rf_model.pkl      ← fitted RF
│   ├── mlp_model.pkl     ← fitted MLP
│   └── scaler.pkl        ← fitted StandardScaler
└── data/
    ├── features_train.csv
    └── features_test.csv
```

### Save models after Phase 5

```python
import joblib
joblib.dump(clf,           "models/rf_model.pkl")
joblib.dump(scaler_final,  "models/scaler.pkl")
joblib.dump(le,            "models/label_encoder.pkl")
```

### agent_core.py

```python
import joblib, numpy as np
from extract_features import extract_features   # your Phase 2 function

rf      = joblib.load("models/rf_model.pkl")
scaler  = joblib.load("models/scaler.pkl")
le      = joblib.load("models/label_encoder.pkl")

EMOTION_EXPLANATIONS = {
    "angry":    "Strong edge gradients detected — consistent with furrowed brow and jaw tension.",
    "happy":    "High brightness + smooth texture — consistent with raised cheeks and Duchenne smile.",
    "neutral":  "Balanced feature profile — no dominant expression signal detected.",
    "sad":      "Dark frame + sharp upper edges — consistent with drooping brow and downturned mouth.",
    "surprise": "High brightness + elevated lower edges — consistent with wide eyes and open mouth.",
}

def predict(img_path: str) -> dict:
    feats     = extract_features(img_path).reshape(1, -1)
    feats_sc  = scaler.transform(feats)
    proba     = rf.predict_proba(feats_sc)[0]
    pred_idx  = np.argmax(proba)
    emotion   = le.classes_[pred_idx]
    return {
        "emotion":     emotion,
        "confidence":  float(proba[pred_idx]),
        "all_probs":   dict(zip(le.classes_, proba.tolist())),
        "explanation": EMOTION_EXPLANATIONS[emotion]
    }
```

### agent_app.py (Streamlit skeleton)

```python
import streamlit as st
from PIL import Image
from agent_core import predict
import tempfile, os

st.title("😊 Facial Expression Classifier")
st.caption("CS513 Spring 2026 — Classical ML Agent Demo")

uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=200)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        result = predict(tmp.name)
        os.unlink(tmp.name)

    st.subheader(f"Predicted: **{result['emotion'].upper()}**")
    st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    st.write("**Why this prediction?**")
    st.info(result["explanation"])

    st.write("**All class probabilities:**")
    st.bar_chart(result["all_probs"])
```

### Run

```bash
streamlit run agent_app.py
```

---

## Project Timeline

| Phase | Task                                                         | Status      |
| ----- | ------------------------------------------------------------ | ----------- |
| 1     | EDA — feature signal validation                             | ✅ Complete |
| 2     | Feature extraction → features_train.csv + features_test.csv | 🔲 Next     |
| 3     | Preprocessing (StandardScaler + SMOTE)                       | 🔲          |
| 4     | Model training (RF, MLP, SVM) + 5-fold CV                    | 🔲          |
| 5     | Final test evaluation + confusion matrix                     | 🔲          |
| 6     | SHAP analysis + per-emotion explanation                      | 🔲          |
| 7     | Streamlit agent demo                                         | 🔲          |

---

## Your Next Step: Phase 2

```
1. Copy extract_features() into 02_features.ipynb
2. Test on ONE image:
      feats = extract_features("data/train/angry/im0.png")
      print(feats.shape)     # (18,)
      print(feats)           # no NaN, reasonable values

3. Run build_csv("train", "features_train.csv")   ← ~5-10 min
4. Run build_csv("test",  "features_test.csv")    ← ~1-2 min
5. Validate with df.describe() — check for NaN, check hog_h00 > hog_h15 in mean
```

---

*CS513 Spring 2026 — Pipeline v2.0 | 18 features*
