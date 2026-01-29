# Emotion Classification (Sentiment Analysis) — 6 Emotions  
**Baseline (TF‑IDF + LinearSVC) + Transformer (DistilBERT Fine‑Tuning)**

This project builds an end-to-end NLP pipeline that classifies a user-written sentence into one of **six emotions**:

**sadness • joy • love • anger • fear • surprise**

Included:
- Dataset loading + sanity checks
- EDA (label distribution, text length analysis)
- Text preprocessing + cleaned datasets
- **Baseline model:** TF‑IDF + LinearSVC
- **Transformer model:** DistilBERT fine‑tuning (CPU‑friendly)
- Validation + Test evaluation (reports + confusion matrices saved to `reports/metrics/`)

---

## Table of Contents
- [1. Dataset](#1-dataset)
- [2. Label Mapping](#2-label-mapping)
- [3. Project Structure](#3-project-structure)
- [4. Method](#4-method)
- [5. Setup](#5-setup)
- [6. Run Pipeline](#6-run-pipeline)
- [7. Results](#7-results)
- [8. Outputs & Artifacts](#8-outputs--artifacts)
- [9. Notes](#9-notes)
- [10. Reproducibility](#10-reproducibility)

---

## 1. Dataset

All CSV files contain two columns:
- `text`: input sentence
- `label`: integer class (0–5)

Splits:
- **Train:** `data/raw/training.csv` (16,000 rows)
- **Validation:** `data/raw/validation.csv` (2,000 rows)
- **Test:** `data/raw/test.csv` (2,000 rows)

> The dataset is **imbalanced** (e.g., `surprise` has fewer samples). Therefore, **Macro F1** is reported alongside Accuracy.

---

## 2. Label Mapping

| Label | Emotion   |
|------:|-----------|
| 0     | sadness   |
| 1     | joy       |
| 2     | love      |
| 3     | anger     |
| 4     | fear      |
| 5     | surprise  |

---

## 3. Project Structure

```text
sentiment-analysis/
  data/
    raw/
      training.csv
      validation.csv
      test.csv
    processed/
      train_clean.csv
      val_clean.csv

  models/
    baseline/
      baseline_tfidf_linearsvc.joblib
    transformer/
      best_model/
        (config/tokenizer/model files)

  notebooks/
    eda.ipynb

  reports/
    figures/
      (EDA plots)
    metrics/
      baseline_classification_report.txt
      baseline_confusion_matrix.png

      transformer_classification_report.txt
      transformer_confusion_matrix.png

      baseline_test_report.txt
      baseline_test_cm.png
      transformer_test_report.txt
      transformer_test_cm.png
      test_summary.txt

    eda_summary.md
    preprocessing.md
    research.md
    results.md

  src/
    sanity_check.py
    preprocess.py
    prepare_processed_data.py
    train_baseline.py
    tune_baseline.py
    train_transformer.py
    evaluate.py

  requirements.txt
  README.md
  .gitignore
```

---

## 4. Method

### 4.1 Preprocessing
A lightweight preprocessing pipeline is applied:
- lowercasing normalization
- whitespace cleanup
- optional digit removal (documented)

Outputs:
- `data/processed/train_clean.csv`
- `data/processed/val_clean.csv`

Details: `reports/preprocessing.md`

### 4.2 Models

**A) Baseline — TF‑IDF + LinearSVC**
- Fast classical ML baseline
- Strong performance on short texts
- Easy to reproduce and deploy

Script: `src/train_baseline.py`

**B) Transformer — DistilBERT Fine‑Tuning**
- Fine‑tunes `distilbert-base-uncased` for 6-class emotion classification
- Better semantic understanding → improved generalization

Script: `src/train_transformer.py`

---

## 5. Setup

### 5.1 Create a virtual environment (Windows PowerShell)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 5.2 Install dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Run Pipeline

### 6.1 Sanity checks (optional)
```bash
python src/sanity_check.py
```

### 6.2 Preprocess (creates `data/processed/*.csv`)
```bash
python src/prepare_processed_data.py
```

### 6.3 EDA (optional)
Open:
- `notebooks/eda.ipynb`

Outputs:
- plots → `reports/figures/`
- EDA notes → `reports/eda_summary.md`

### 6.4 Train Baseline
```bash
python src/train_baseline.py
```

Outputs:
- model → `models/baseline/baseline_tfidf_linearsvc.joblib`
- validation report + confusion matrix → `reports/metrics/`

(Optional) Baseline tuning:
```bash
python src/tune_baseline.py
```

### 6.5 Train Transformer (DistilBERT)
```bash
python src/train_transformer.py
```

Outputs:
- model → `models/transformer/best_model/`
- validation report + confusion matrix → `reports/metrics/transformer_*`

### 6.6 Evaluate on Test (Baseline + Transformer)
```bash
python src/evaluate.py
```

Outputs (saved):
- `reports/metrics/baseline_test_report.txt`
- `reports/metrics/baseline_test_cm.png`
- `reports/metrics/transformer_test_report.txt`
- `reports/metrics/transformer_test_cm.png`
- `reports/metrics/test_summary.txt`

---

## 7. Results

### 7.1 Validation
| Model | Accuracy | Macro F1 |
|------|---------:|---------:|
| Baseline (TF‑IDF + LinearSVC) | **0.9030** | **0.8735** |
| Transformer (DistilBERT, 1 epoch) | **0.9255** | **0.8987** |

### 7.2 Test (Final)
| Model | Accuracy | Macro F1 |
|------|---------:|---------:|
| Baseline (TF‑IDF + LinearSVC) | **0.8870** | **0.8395** |
| Transformer (DistilBERT, 1 epoch) | **0.9235** | **0.8855** |

---

## 8. Outputs & Artifacts

### Saved models
- Baseline: `models/baseline/baseline_tfidf_linearsvc.joblib`
- Transformer: `models/transformer/best_model/`

### Metrics & plots
All metrics and plots are saved under:
- `reports/metrics/` (classification reports + confusion matrices for val/test)
- `reports/figures/` (EDA figures)

Key files:
- Validation:
  - `reports/metrics/baseline_classification_report.txt`
  - `reports/metrics/baseline_confusion_matrix.png`
  - `reports/metrics/transformer_classification_report.txt`
  - `reports/metrics/transformer_confusion_matrix.png`
- Test:
  - `reports/metrics/baseline_test_report.txt`
  - `reports/metrics/baseline_test_cm.png`
  - `reports/metrics/transformer_test_report.txt`
  - `reports/metrics/transformer_test_cm.png`
  - `reports/metrics/test_summary.txt`

---

## 9. Notes

- **Macro F1** is important due to class imbalance (minority emotions such as `surprise`).
- Confusions commonly observed (see confusion matrices):
  - **joy ↔ love** (overlapping positive language)
  - **fear ↔ surprise** (short/ambiguous sentences)

Additional documentation:
- Research notes: `reports/research.md`
- Results interpretation: `reports/results.md`

---

## 10. Reproducibility

- Fixed random seed where applicable (e.g., `seed=42` in training scripts).
- Scripts save outputs to deterministic locations under `models/` and `reports/metrics/`.
- Recommended run order:
  1) `python src/prepare_processed_data.py`
  2) `python src/train_baseline.py`
  3) `python src/train_transformer.py`
  4) `python src/evaluate.py`

## Demo UI (Gradio) Optional
To demonstrate the trained Transformer model in an interactive way, a simple **Gradio web interface** is provided.

The interface allows a user to:
- Enter any custom sentence
- Get the **predicted emotion**
- See **class probabilities** for all six emotions
- Test the model with predefined example sentences

This satisfies the optional requirement of the task to validate the model using a simple UI.

Make sure the Transformer model has been trained and saved to:
Then run:
```bash
python app/gradio_app.py
```
After running the command, Gradio will start a local server.

Open the URL printed in the terminal (usually): http://127.0.0.1:7860

Model: DistilBERT (fine-tuned)
Path: models/transformer/best_model/



## Pretrained Model & Data (External Download)

Due to GitHub file size limitations, large files are shared externally.

### Transformer Model
- Download (OneDrive): <PUT_LINK_HERE>

After downloading, extract the folder to:
models/transformer/best_model/

### (Optional) Processed Data
- Download (OneDrive): <PUT_LINK_HERE>

Expected structure after extraction:
models/
  transformer/
    best_model/
