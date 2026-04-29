# CS513 Final Project: Facial Emotion Detection

A comprehensive data mining pipeline for facial emotion classification using the FER-2013 dataset. This project implements custom feature extraction, model training, and evaluation with a Gradio web application for inference.

## Project Overview

This is a machine learning project for classifying facial expressions into **5 emotion categories**: angry, happy, neutral, sad, and surprise. The system uses classical computer vision features combined with ensemble and deep learning models.

**Best Model Performance**: Random Forest with **62.52% test accuracy** and **61.44% macro-F1 score**

---

## Dataset

- **Source**: FER-2013 (Facial Expression Recognition Dataset)
- **Train Split**: 28,821 images
- **Test Split**: 7,178 images
- **Format**: 48×48 pixel grayscale images
- **Classes**: 5 emotions (angry, happy, neutral, sad, surprise)
- **Challenge**: 2.27× class imbalance (balanced with SMOTE during training)

---

## Feature Engineering

The project extracts **18 core features** per 48×48 image:

| Feature Family | Count | Description | Signal Strength |
|---|---|---|---|
| **Intensity** | 1 | Brightness mean | 1.20× moderate |
| **Edge Detection** | 1 | Lower face edge density (mouth/jaw activity) | 1.11× moderate |
| **HOG Histogram** | 16 | 16-bin Histogram of Oriented Gradients | 1.81× strong (bins 13–15) |

**Feature Signal Summary**:
- ✅ **Strong** (>1.3×): 3 features (captures high-gradient regions)
- ⚠️ **Moderate** (1.1–1.3×): 6 features
- ❌ **Weak** (<1.1×): 9 features (kept for Random Forest interaction modeling)

*Optional Extension*: 17 additional facial geometry features via MediaPipe FaceMesh (F25–F41 in pipeline_v2_F41.md)

---

## Model Comparison

### Test Set Performance

| Model | Accuracy | Macro-F1 | Training Time |
|---|---|---|---|
| **Random Forest** ⭐ | **62.52%** | **61.44%** | 2.8s |
| SVM (RBF) | 58.33% | 57.23% | 181.7s |
| MLP/ANN | 54.89% | 53.75% | 25.5s |
| Decision Tree | 50.60% | 49.98% | 1.8s |
| KNN | 51.02% | 50.88% | 0.0s |
| Naive Bayes | 51.38% | 49.03% | 0.0s |

### Per-Emotion Performance (Random Forest)

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Happy | 0.827 | 0.749 | **0.786** | 1,774 |
| Surprise | 0.732 | 0.743 | **0.737** | 831 |
| Neutral | 0.484 | 0.652 | 0.556 | 1,233 |
| Angry | 0.525 | 0.464 | 0.493 | 958 |
| Sad | 0.538 | 0.469 | 0.501 | 1,247 |

---

## Project Pipeline

The main workflow for this project follows the F41 notebook set:

```
Phase 1: Feature Extraction
  └─ notebooks/01_features_F41.ipynb → notebooks/csv/features_train_F41.csv, notebooks/csv/features_test_F41.csv
  
Phase 2: Exploratory Data Analysis (EDA)
  └─ notebooks/02_eda_F41.ipynb
  
Phase 3: Model Training & Evaluation
  └─ notebooks/03_train_F41.ipynb → model_comparison_F41.csv
  
Phase 4: Model Interpretation
  └─ notebooks/04_evaluate_F41.ipynb → SHAP importance analysis
  
Phase 5: Web Application
  └─ app/ (Gradio interface and inference entrypoint)
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages**:
- scikit-learn, scikit-image (ML & feature extraction)
- opencv-python (image processing)
- mediapipe (facial landmarks)
- gradio (web app)
- pandas, numpy (data manipulation)
- shap (model interpretation)

### 2. Run Gradio App

```bash
python -m app.app
```

Then open the local Gradio URL shown in the terminal.

**App Features**:
- ✅ Upload facial images (JPEG, PNG)
- ✅ Real-time emotion classification
- ✅ Confidence score display
- ✅ JSON report export
- ✅ Batch processing support

### 3. Run Training Pipeline

To retrain models from scratch:

```bash
# Step 1: Extract features from raw images
jupyter nbconvert --to notebook --execute notebooks/01_features_F41.ipynb

# Step 2: Train models (5-fold CV) and generate reports
jupyter nbconvert --to notebook --execute notebooks/03_train_F41.ipynb

# Step 3: Generate SHAP interpretation plots
jupyter nbconvert --to notebook --execute notebooks/04_evaluate_F41.ipynb
```

---

## Project Structure

```
CS513-Final-Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── app/
│   ├── __init__.py                    # App package
│   ├── app.py                         # Gradio web application entrypoint
│   └── inference.py                   # Model inference utilities
├── pipeline_v2.md                     # Detailed methodology and feature pipeline
│
├── data/
│   ├── train/                         # 28,821 training images
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                          # 7,178 test images (same structure)
│
├── notebooks/
│   ├── 01_features_F41.ipynb          # Feature extraction & validation
│   ├── 02_eda_F41.ipynb               # Exploratory data analysis
│   ├── 03_train_F41.ipynb             # Model training (6 classifiers, 5-fold CV)
│   ├── 04_evaluate_F41.ipynb          # Evaluation & SHAP analysis
│   ├── csv/
│   │   ├── features_train_F41.csv     # Extracted features (train)
│   │   └── features_test_F41.csv      # Extracted features (test)
│   └── output/
│       ├── 02_eda/                    # EDA visualizations & statistics
│       ├── 03_train/
│       │   └── model_comparison_F41.csv     # Test accuracy/F1 for all models
│       └── 04_evaluate/
│           ├── classification_report_F41.csv  # Per-class metrics
│           └── shap_importances_F41.csv       # Feature importance scores
```

---

## Methodology & Key Findings

### Data Preprocessing
1. **Image standardization**: All images resized to 48×48 grayscale (no further resizing needed)
2. **Label encoding**: 5 emotions → class indices 0–4
3. **Class balancing**: SMOTE applied within each 5-fold CV training fold
4. **Feature scaling**: StandardScaler (0-mean, unit variance)

### Cross-Validation Strategy
- **Method**: 5-Fold Stratified K-Fold
- **Rationale**: Ensures balanced class distribution per fold and reduces variance in metric estimates
- **SMOTE application**: Applied only to training fold (not validation or test)

### Model Selection
- **Random Forest** (best): Captures feature interactions; provides SHAP importances for interpretability
- **SVM (RBF)**: Strong baseline on the compact F24 base-feature space
- **MLP**: Explores non-linear feature combinations
- **Baseline models**: KNN, Decision Tree, Naive Bayes for comparison

### Feature Insights
- **HOG dominates**: Bins 13–15 show 1.50–1.81× class separability — captures high-gradient facial regions (muscle tension indicative of anger/sadness)
- **Brightness matters**: 1.20× separability — surprise has higher brightness
- **Weak features kept**: In Random Forest for interaction effects (9 weak features contribute through combinations)

### Performance Analysis
- **Best performers**: Happy (79% F1) and Surprise (74% F1) — distinct HOG patterns
- **Challenging classes**: Angry and Sad (49–50% F1) — similar facial muscle activation
- **Imbalance effect**: SMOTE successfully mitigates initial 2.27× imbalance

---

## Feature Details

### 41-Feature Set (F01–F41)

The current extractor uses 41 features per image: 24 base visual features and 17 facial-geometry features.

**Base Features (F01–F24)**:
- F01: `brightness_mean` — normalized mean pixel intensity
- F02: `lower_edge_density` — Canny edge density in the lower face region
- F03–F18: `hog_h00`–`hog_h15` — 16-bin Histogram of Oriented Gradients
- F19–F24: `lbp_h00`–`lbp_h05` — 6-bin Local Binary Pattern histogram

**Geometry Features (F25–F41)**:
- F25: `ear_L` — left eye aspect ratio
- F26: `ear_R` — right eye aspect ratio
- F27: `brow_eye_L` — left brow-to-eye distance
- F28: `brow_eye_R` — right brow-to-eye distance
- F29: `inter_brow` — distance between the brows
- F30: `brow_raise_L` — left brow raise
- F31: `brow_raise_R` — right brow raise
- F32: `MAR` — mouth aspect ratio
- F33: `lip_pull` — lip corner pull / smile stretch
- F34: `lip_droop` — lip corner droop
- F35: `mouth_curl` — mouth curl / upward turn
- F36: `lip_tight` — lip tightness
- F37: `jaw_open` — jaw openness
- F38: `cheek_L` — left cheek raise
- F39: `cheek_R` — right cheek raise
- F40: `nose_lip` — nose-to-lip distance
- F41: `face_AR` — face aspect ratio

**Feature Notes**:
- HOG remains the strongest base signal for local facial structure.
- LBP adds micro-texture sensitivity for subtle expression cues.
- Geometry features are normalized with facial proportions so the F41 set stays image-scale aware.
- The resulting training CSVs contain 42 columns total: 41 features plus the `label` column.

---

## Results & Reproducibility

All model results are stored in `notebooks/output/`:

- **Training metrics**: `03_train/model_comparison_F41.csv`
- **Per-class evaluation**: `04_evaluate/classification_report_F41.csv`
- **Feature importance**: `04_evaluate/shap_importances_F41.csv`

To reproduce:
1. Ensure FER-2013 data is in `data/train/` and `data/test/`
2. Run `notebooks/01_features_F41.ipynb` to extract features
3. Execute notebooks in order: 02 → 03 → 04

---

## References & Literature

- **FER-2013 Dataset**: [Goodfellow et al., 2013](https://arxiv.org/abs/1307.0414)
- **HOG Features**: Dalal & Triggs, 2005 — Histograms of Oriented Gradients
- **Random Forest**: Breiman, 2001
- **SMOTE**: Chawla et al., 2002 — handling imbalanced datasets
- **SHAP**: Lundberg & Lee, 2017 — model interpretability

---

## Authors

CS513 Data Mining Final Project | Spring 2026

---

## Notes

- OpenCV Canny edge detection requires CV2; ensure `opencv-python` ≥ 4.9.0
- MediaPipe face detection requires ≥ 0.10; face mesh FaceMesh crashes on images without detectable faces (graceful fallback to NaN)
- SHAP computations can be slow (>5min for 300 trees); consider sampling features for large models
- Best performance on frontal face images; side profiles may reduce accuracy