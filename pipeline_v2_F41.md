# Facial Expression Classification

## Pipeline & Implementation Plan v2 (F41 + Geometry)

> **Feature set:** F41 = 24 original (brightness, edge, HOG×16, LBP×6) + 17 facial-geometry features via MediaPipe FaceMesh
> **Best model so far:** RandomForest — test acc 0.3703, macro-F1 0.3597 (F24 baseline)
> **Target with geometry:** macro-F1 ≥ 0.42 (literature: +10–20% from geometry + HOG)

---

## Quick Reference

| Phase         | Notebook                  | Input        | Output                     | Status    |
| ------------- | ------------------------- | ------------ | -------------------------- | --------- |
| 1 — EDA      | `01_eda_r3.ipynb`       | FER-2013 raw | class-dist charts          | ✅ Done   |
| 2 — Features | `02_features_r1.ipynb`  | images       | `features_train_F41.csv` | 🔄 Update |
| 3 — Train    | `03_train_r1_F41.ipynb` | F41 CSV      | `results_F41.csv`        | ⏳ Next   |
| 4 — SHAP     | `04_shap_r1.ipynb`      | RF model     | SHAP plots                 | ⏳ Next   |
| 5 — Agent    | `05_agent_demo.py`      | RF + scaler  | Streamlit app              | ⏳ Next   |

---

## Phase 1 — Exploratory Data Analysis ✅

Already complete (`01_eda_r3.ipynb`). Key findings carried forward:

- **5 classes:** angry 3995, happy 7215, neutral 4965, sad 4830, surprise 3171
- Imbalance ratio: 2.28× → SMOTE required in Phase 3
- HOG signal ratio 1.81× (strongest) — `hog_h14`, `hog_h15` most discriminative
- LBP signal ratio 1.18× (weakest) — JPEG artifacts wash out micro-texture at 48×48
- Geometry signal: EAR, MAR, brow distances expected 1.4–1.9× (literature-backed)

---

## Phase 2 — Feature Extraction (Updated → F41)

### 2.1 Install dependencies

```bash
pip install mediapipe opencv-python-headless scikit-image numpy pandas tqdm
```

### 2.2 Feature families

| ID                 | Features                    | Count        | Extractor                                  | Notes                                |
| ------------------ | --------------------------- | ------------ | ------------------------------------------ | ------------------------------------ |
| F01                | `brightness_mean`         | 1            | `np.mean(gray)`                          | Normalised 0–1                      |
| F02                | `lower_edge_density`      | 1            | Canny on bottom 60% of face                | Jaw/mouth activity                   |
| F03–F18           | `hog_h00`–`hog_h15`    | 16           | `skimage.feature.hog`                    | 16-bin global histogram, 11.25°/bin |
| F19–F24           | `lbp_h00`–`lbp_h05`    | 6            | `skimage.feature.local_binary_pattern`   | P=8 R=1 uniform, 6-bucket            |
| **F25–F41** | **geometry features** | **17** | **MediaPipe FaceMesh 468 landmarks** | **NEW — see §2.4**           |
|                    | **Total**             | **41** |                                            |                                      |

### 2.3 Base feature extractor (unchanged from r1)

```python
import cv2, numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_base_features(gray):
    """F01–F24: brightness, edge, HOG, LBP"""
    # F01 — brightness
    brightness = gray.mean() / 255.0

    # F02 — lower edge density (bottom 60% of face)
    h = gray.shape[0]
    lower = gray[int(0.4 * h):]
    edges = cv2.Canny(lower, threshold1=30, threshold2=80)
    lower_edge = edges.mean() / 255.0

    # F03–F18 — HOG (16-bin global histogram)
    _, hog_img = hog(gray, orientations=16, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1), visualize=True)
    from skimage.exposure import rescale_intensity
    hog_img = rescale_intensity(hog_img, in_range=(0, 10))
    hog_hist, _ = np.histogram(hog_img.ravel(), bins=16, range=(0, 1), density=True)
    hog_hist = hog_hist / (hog_hist.sum() + 1e-8)

    # F19–F24 — LBP (6-bin uniform)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=6, range=(0, 9), density=True)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)

    return [brightness, lower_edge] + list(hog_hist) + list(lbp_hist)
```

### 2.4 Geometry feature extractor (NEW — F25–F41)

```python
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
_MESH   = mp_face.FaceMesh(
    static_image_mode        = True,
    max_num_faces            = 1,
    refine_landmarks         = True,
    min_detection_confidence = 0.5,
)

def _pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])

def _dist(lm, a, b, w, h):
    return np.linalg.norm(_pt(lm, a, w, h) - _pt(lm, b, w, h))

def _ear(lm, p1, p2, p3, p4, p5, p6, w, h):
    """Eye Aspect Ratio: vertical / horizontal span."""
    v = _dist(lm, p2, p6, w, h) + _dist(lm, p3, p5, w, h)
    return v / (2.0 * _dist(lm, p1, p4, w, h) + 1e-6)

def extract_geometry(gray):
    """F25–F41: 17 MediaPipe landmark-derived features.
    Returns list of 17 floats, or [np.nan]*17 if detection fails.
    All distances normalised by inter-ocular distance (IOD).
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    res = _MESH.process(rgb)
    if not res.multi_face_landmarks:
        return [np.nan] * 17

    lm = res.multi_face_landmarks[0].landmark
    h, w = gray.shape

    iod = _dist(lm, 33, 263, w, h) + 1e-6   # inter-ocular distance

    return [
        # ── Eyes ──────────────────────────────────────────────────────
        _ear(lm, 33,  160, 158, 133, 153, 144, w, h),  # F25 EAR_left
        _ear(lm, 362, 385, 387, 263, 373, 380, w, h),  # F26 EAR_right

        # ── Brows ─────────────────────────────────────────────────────
        (_pt(lm, 70, w, h)[1]  - _pt(lm, 159, w, h)[1]) / iod,  # F27 brow_eye_dist_L
        (_pt(lm, 300, w, h)[1] - _pt(lm, 386, w, h)[1]) / iod,  # F28 brow_eye_dist_R
        _dist(lm, 55, 285, w, h) / iod,                          # F29 inter_brow_dist
        (_pt(lm, 70,  w, h)[1] - _pt(lm, 105, w, h)[1]) / iod,  # F30 brow_raise_L
        (_pt(lm, 300, w, h)[1] - _pt(lm, 334, w, h)[1]) / iod,  # F31 brow_raise_R

        # ── Mouth ─────────────────────────────────────────────────────
        _dist(lm, 13, 14, w, h) / (                              # F32 MAR (mouth aspect ratio)
            _dist(lm, 61, 291, w, h) + 1e-6),
        _dist(lm, 61, 291, w, h) / iod,                          # F33 lip_corner_pull
        (_pt(lm, 61,  w, h)[1] + _pt(lm, 291, w, h)[1])         # F34 lip_corner_droop
            / (2 * h),
        (_pt(lm, 13,  w, h)[1] - _pt(lm, 14,  w, h)[1]) / iod,  # F35 mouth_curl_up
        _dist(lm, 82, 87, w, h) / iod,                           # F36 lip_tightness
        _dist(lm, 152, 1, w, h) / iod,                           # F37 jaw_openness

        # ── Cheeks ────────────────────────────────────────────────────
        (_pt(lm, 116, w, h)[1] - _pt(lm, 93,  w, h)[1]) / iod,  # F38 cheek_raise_L
        (_pt(lm, 345, w, h)[1] - _pt(lm, 323, w, h)[1]) / iod,  # F39 cheek_raise_R

        # ── Nose / Global ─────────────────────────────────────────────
        _dist(lm, 94, 13, w, h) / iod,                           # F40 nose_lip_dist
        h / (w + 1e-6),                                           # F41 face_aspect_ratio
    ]
```

### 2.5 Combined extractor and CSV builder

```python
FEATURE_NAMES = (
    ["brightness_mean", "lower_edge_density"]
    + [f"hog_h{i:02d}" for i in range(16)]
    + [f"lbp_h{i:02d}" for i in range(6)]
    + ["ear_L", "ear_R",
       "brow_eye_L", "brow_eye_R", "inter_brow",
       "brow_raise_L", "brow_raise_R",
       "MAR", "lip_pull", "lip_droop", "mouth_curl",
       "lip_tight", "jaw_open",
       "cheek_L", "cheek_R",
       "nose_lip", "face_AR"]
)  # len = 41

def extract_features(gray):
    """Returns list of 41 floats for one 48×48 grayscale image."""
    return extract_base_features(gray) + extract_geometry(gray)

def build_csv(split_dir, out_csv, max_per_class=None):
    """Walk split_dir/emotion/img.jpg → features_F41.csv"""
    import os
    from tqdm import tqdm

    records = []
    emotions = sorted(os.listdir(split_dir))
    for emo in emotions:
        imgs = sorted(os.listdir(os.path.join(split_dir, emo)))
        if max_per_class:
            imgs = imgs[:max_per_class]
        for fname in tqdm(imgs, desc=emo):
            path = os.path.join(split_dir, emo, fname)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, (48, 48))
            feats = extract_features(gray)
            records.append(feats + [emo])

    import pandas as pd
    df = pd.DataFrame(records, columns=FEATURE_NAMES + ["label"])

    # Handle NaN from missed MediaPipe detections (fill with column median)
    nan_count = df[FEATURE_NAMES].isna().sum().sum()
    if nan_count > 0:
        print(f"  ⚠ {nan_count} NaN values from missed detections → filling with median")
        df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())

    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}  shape={df.shape}")
    return df

# Run:
# df_train = build_csv("data/train", "features_train_F41.csv")
# df_test  = build_csv("data/test",  "features_test_F41.csv")
```

### 2.6 Validation checks

```python
# After building CSVs, run these assertions:
import pandas as pd
df = pd.read_csv("features_train_F41.csv")

assert df.shape[1] == 42,            f"Expected 42 cols (41 features + label), got {df.shape[1]}"
assert df.isna().sum().sum() == 0,   "NaN values remain after median fill"
assert set(df["label"]) == {"angry","happy","neutral","sad","surprise"}
assert df[df["label"]=="happy"].shape[0] > 7000,  "happy class count mismatch"

# Check geometry columns are non-trivial (std > 0)
geo_cols = ["ear_L","ear_R","brow_eye_L","MAR","jaw_open","lip_pull","inter_brow"]
for col in geo_cols:
    sig = (df.groupby("label")[col].mean().max() /
           df.groupby("label")[col].mean().min())
    print(f"  {col:<18} signal ratio = {sig:.3f}x")
    # Expect ≥ 1.3× for geometry to be useful
print("Validation passed ✅")
```

### 2.7 Expected geometry signal ratios

| Feature          | Emotion pair (max/min mean) | Expected ratio | Why                     |
| ---------------- | --------------------------- | -------------- | ----------------------- |
| `MAR`          | surprise vs neutral         | 2.1×          | Open mouth vs closed    |
| `jaw_open`     | surprise vs angry           | 1.9×          | Wide jaw vs clenched    |
| `ear_L/R`      | surprise vs sad             | 1.7×          | Wide eyes vs squinted   |
| `brow_eye_L/R` | surprise vs angry           | 1.6×          | Raised vs furrowed brow |
| `inter_brow`   | neutral vs angry            | 1.5×          | Brows apart vs pinched  |
| `lip_pull`     | happy vs sad                | 1.4×          | Smile stretch vs droop  |
| `cheek_L/R`    | happy vs neutral            | 1.3×          | Cheek puff from smile   |

---

## Phase 3 — Preprocessing & Training (Updated → F41)

### 3.1 Experiment design

| Experiment        | Features                       | Note                                      |
| ----------------- | ------------------------------ | ----------------------------------------- |
| A (baseline)      | F24 original                   | Already done in `03_train_r0_F24.ipynb` |
| **B (new)** | **F41 = F24 + geometry** | Run `03_train_r1_F41.ipynb`             |
| C (ablation)      | F24 − LBP + geometry          | Swap LBP for geometry, same count (F35)   |

### 3.2 Pipeline (no changes from r0)

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

pipeline = ImbPipeline([
    ("smote",  SMOTE(random_state=42, k_neighbors=5)),
    ("scaler", StandardScaler()),
    ("clf",    clf),   # swap in each model below
])
```

> ⚠ Geometry features like `MAR` and `jaw_open` have very different scales from HOG bins (which are 0–1 normalised densities). StandardScaler is **critical** — do not skip it.

### 3.3 Models and expected performance

| Model        | F24 macro-F1     | F41 estimate          | Change             |
| ------------ | ---------------- | --------------------- | ------------------ |
| **RF** | **0.3597** | **~0.42–0.46** | **+17–28%** |
| SVM (RBF)    | 0.3203           | ~0.38–0.42           | +19–31%           |
| MLP          | 0.3049           | ~0.36–0.40           | +18–31%           |
| KNN          | 0.3017           | ~0.33–0.37           | +9–23%            |
| DT           | 0.2896           | ~0.31–0.34           | +7–17%            |
| NB           | 0.2801           | ~0.29–0.32           | +4–14%            |

### 3.4 Hyperparameter tuning (RF priority)

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "clf__n_estimators"    : [100, 200, 400],
    "clf__max_features"    : ["sqrt", "log2", 0.3],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__criterion"       : ["entropy", "gini"],
}
search = RandomizedSearchCV(
    pipeline, param_grid,
    n_iter=20, cv=5,
    scoring="f1_macro",
    n_jobs=-1, random_state=42, verbose=2,
)
search.fit(X_train, y_train)
print(search.best_params_)
```

### 3.5 Cross-validation scoring

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, X_train, y_train, cv=cv,
    scoring={"acc": "accuracy", "f1": "f1_macro"},
    return_train_score=False, n_jobs=-1,
)
print(f"CV acc  = {cv_results['test_acc'].mean():.4f} ± {cv_results['test_acc'].std():.4f}")
print(f"CV F1   = {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
```

---

## Phase 4 — SHAP Explainability (Updated → F41)

### 4.1 Changes from r0

- Replace `FEATURES` list: now 41 names (use `FEATURE_NAMES` from Phase 2)
- Add format-safe SHAP output handling (fixes ValueError from Phase 4 r0):

```python
# ── SHAP format-safe aggregation (fixes r0 bug) ──────────────────────
shap_values = explainer.shap_values(X_shap)

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_arr  = shap_values.transpose(2, 0, 1)        # (5, n, 41)
else:
    shap_arr  = np.stack(shap_values, axis=0)          # (5, n, 41)

shap_list = [shap_arr[i] for i in range(5)]            # list of 5 arrays
mean_abs  = np.abs(shap_arr).mean(axis=(0, 1))         # (41,) ✅
```

### 4.2 Expected SHAP ranking changes (F24 → F41)

```
F24 ranking:          F41 expected ranking:
  #1 hog_h15            #1 MAR              ← geometry takes top spots
  #2 hog_h14            #2 jaw_open
  #3 brightness_mean    #3 ear_L / ear_R
  #4 lower_edge         #4 hog_h15
  #5 hog_h13            #5 brow_eye_L/R
  …                     #6 brightness_mean
  #23 lbp_h03           …
  #24 lbp_h05           #38–41 lbp bins (still bottom)
```

### 4.3 New SHAP plot: geometry vs HOG contribution

```python
# Split features into groups and compare group-level SHAP
groups = {
    "HOG"      : [i for i, f in enumerate(FEATURE_NAMES) if f.startswith("hog")],
    "LBP"      : [i for i, f in enumerate(FEATURE_NAMES) if f.startswith("lbp")],
    "Base"     : [0, 1],
    "Geometry" : list(range(24, 41)),
}
group_shap = {g: np.abs(shap_arr).mean(axis=(0,1))[idx].sum()
              for g, idx in groups.items()}

import pandas as pd
gs_df = pd.DataFrame(list(group_shap.items()), columns=["group","total_shap"])
gs_df = gs_df.sort_values("total_shap", ascending=False)
print(gs_df)   # Expected: Geometry > HOG > Base > LBP
```

---

## Phase 5 — Streamlit Agent Demo (Updated)

### 5.1 Architecture

```
User uploads image
      │
      ▼
cv2.cvtColor → 48×48 grayscale
      │
      ├─── extract_base_features()  → 24 floats (HOG, LBP, etc.)
      │
      └─── extract_geometry()       → 17 floats (MediaPipe)
                │
                ▼
           41-feature vector
                │
                ▼
       StandardScaler.transform()
                │
                ▼
        RF.predict_proba()  →  top-2 emotions + confidence
                │
                ▼
        shap.TreeExplainer  →  waterfall plot (top-8 features)
                │
                ▼
        Streamlit display
```

### 5.2 Minimal Streamlit skeleton

```python
# 05_agent_demo.py
import streamlit as st
import joblib, cv2, numpy as np, shap, matplotlib.pyplot as plt

@st.cache_resource
def load_artifacts():
    rf     = joblib.load("rf_f41.pkl")
    scaler = joblib.load("scaler_f41.pkl")
    explainer = shap.TreeExplainer(rf)
    return rf, scaler, explainer

EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]

st.title("😊 Facial Expression Classifier — F41 Features")
uploaded = st.file_uploader("Upload a face image", type=["jpg","png","jpeg"])

if uploaded:
    buf   = np.frombuffer(uploaded.read(), np.uint8)
    img   = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    gray  = cv2.resize(img, (48, 48))

    rf, scaler, explainer = load_artifacts()

    feats  = np.array(extract_features(gray)).reshape(1, -1)
    feats_sc = scaler.transform(feats)

    proba  = rf.predict_proba(feats_sc)[0]
    top2   = np.argsort(proba)[::-1][:2]

    st.image(gray, caption="Input (48×48)", width=150)
    col1, col2 = st.columns(2)
    col1.metric(f"#{1} {EMOTIONS[top2[0]]}", f"{proba[top2[0]]:.1%}")
    col2.metric(f"#{2} {EMOTIONS[top2[1]]}", f"{proba[top2[1]]:.1%}")

    # SHAP waterfall
    sv = explainer.shap_values(feats_sc)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv_pred = sv[:, 0, top2[0]]
    else:
        sv_pred = sv[top2[0]][0]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(values=sv_pred,
                         base_values=explainer.expected_value[top2[0]],
                         feature_names=FEATURE_NAMES),
        max_display=8, show=False)
    st.pyplot(fig)
```

### 5.3 Run commands

```bash
# Save fitted artifacts after Phase 3
import joblib
joblib.dump(rf,     "rf_f41.pkl")
joblib.dump(scaler, "scaler_f41.pkl")

# Launch app
streamlit run 05_agent_demo.py
```

---

## Dependency List (full)

```txt
# requirements.txt
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scikit-image>=0.21
imbalanced-learn>=0.11
opencv-python-headless>=4.8
mediapipe>=0.10
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.30
joblib>=1.3
tqdm>=4.65
```

---

## File Map

```
cs513-facial/
├── data/
│   ├── train/
│   │   ├── angry/     (3995 images)
│   │   ├── happy/     (7215 images)
│   │   ├── neutral/   (4965 images)
│   │   ├── sad/       (4830 images)
│   │   └── surprise/  (3171 images)
│   └── test/          (same structure, ~6043 total)
│
├── features_train_F41.csv     ← Phase 2 output (41 features + label)
├── features_test_F41.csv      ← Phase 2 output
├── shap_global_importance.csv ← Phase 4 output
│
├── rf_f41.pkl                 ← Phase 3 fitted model
├── scaler_f41.pkl             ← Phase 3 fitted scaler
│
├── 01_eda_F41.ipynb  
├── 02_features_F41.ipynb   
├── 03_train_F41.ipynb   
├── 04_shap_F41.ipynb   
└── 05_agent_demo.py   
```

---
