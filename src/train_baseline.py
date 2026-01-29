from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

LABEL2EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

DATA_DIR = Path("data/processed")
TRAIN_PATH = DATA_DIR / "train_clean.csv"
VAL_PATH = DATA_DIR / "val_clean.csv"

MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(int)
    X_val, y_val = val_df["text"].astype(str), val_df["label"].astype(int)

    #TF-IDF + LinearSVC
    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )),
            ("svm", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Macro F1: {f1_macro:.4f}\n")

    #report
    report = classification_report(
        y_val, y_pred,
        target_names=[LABEL2EMOTION[i] for i in sorted(LABEL2EMOTION.keys())],
        digits=4
    )
    print(report)

    (REPORT_DIR / "baseline_classification_report.txt").write_text(report, encoding="utf-8")

    #Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=sorted(LABEL2EMOTION.keys()))
    disp = ConfusionMatrixDisplay(cm, display_labels=[LABEL2EMOTION[i] for i in sorted(LABEL2EMOTION.keys())])

    plt.figure()
    disp.plot(values_format="d")
    plt.title("Baseline Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "baseline_confusion_matrix.png", dpi=200)
    plt.show()

    # Save model
    joblib.dump(clf, MODEL_DIR / "baseline_tfidf_linearsvc.joblib")
    print(f"\nSaved model to: {MODEL_DIR / 'baseline_tfidf_linearsvc.joblib'}")


if __name__ == "__main__":
    main()
