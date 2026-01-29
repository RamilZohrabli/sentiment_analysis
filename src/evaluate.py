from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2EMOTION = {0:"sadness",1:"joy",2:"love",3:"anger",4:"fear",5:"surprise"}
EMOTIONS_ORDER = [LABEL2EMOTION[i] for i in sorted(LABEL2EMOTION.keys())]
TEST_PATH = Path("data/raw/test.csv")
REPORT_DIR = Path("reports/metrics")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_PATH = Path("models/baseline/baseline_tfidf_linearsvc.joblib")
TRANSFORMER_DIR = Path("models/transformer/best_model")

def save_cm(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(LABEL2EMOTION.keys()))
    disp = ConfusionMatrixDisplay(cm, display_labels=EMOTIONS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def eval_baseline(df):
    model = joblib.load(BASELINE_PATH)
    y_true = df["label"].astype(int).values
    y_pred = model.predict(df["text"].astype(str).values)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, target_names=EMOTIONS_ORDER, digits=4)

    (REPORT_DIR / "baseline_test_report.txt").write_text(rep, encoding="utf-8")
    save_cm(y_true, y_pred, "Baseline Confusion Matrix (Test)", REPORT_DIR / "baseline_test_cm.png")

    print("\nBASELINE TEST:", {"accuracy": acc, "macro_f1": f1m})
    return acc, f1m

@torch.no_grad()
def eval_transformer(df):
    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR))
    model.eval()

    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).values

    batch_size = 32
    preds_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_all.append(preds)

    y_pred = np.concatenate(preds_all)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, target_names=EMOTIONS_ORDER, digits=4)

    (REPORT_DIR / "transformer_test_report.txt").write_text(rep, encoding="utf-8")
    save_cm(y_true, y_pred, "Transformer Confusion Matrix (Test)", REPORT_DIR / "transformer_test_cm.png")

    print("\nTRANSFORMER TEST:", {"accuracy": acc, "macro_f1": f1m})
    return acc, f1m

def main():
    df = pd.read_csv(TEST_PATH)
    print("Test shape:", df.shape)

    b_acc, b_f1 = eval_baseline(df)
    t_acc, t_f1 = eval_transformer(df)

    summary = (
        f"BASELINE  - acc={b_acc:.4f} macro_f1={b_f1:.4f}\n"
        f"TRANSFORM - acc={t_acc:.4f} macro_f1={t_f1:.4f}\n"
    )
    (REPORT_DIR / "test_summary.txt").write_text(summary, encoding="utf-8")
    print("\nSaved:", REPORT_DIR / "test_summary.txt")
    print(summary)

if __name__ == "__main__":
    main()
