from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

TRAIN_PATH = Path("data/processed/train_clean.csv")
VAL_PATH   = Path("data/processed/val_clean.csv")

def eval_one(name, pipe, X_train, y_train, X_val, y_val):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, pred)
    f1m = f1_score(y_val, pred, average="macro")
    print(f"{name:22s}  acc={acc:.4f}  macro_f1={f1m:.4f}")
    return f1m, acc, pipe

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(int)
    X_val, y_val = val_df["text"].astype(str), val_df["label"].astype(int)

    candidates = [
        ("word_1_2_min2", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ])),
        ("word_1_3_min2", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=2, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ])),
        ("char_3_5_min2", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ])),
        ("char_4_6_min2", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(4,6), min_df=2, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ])),
    ]

    best = None  # (f1, acc, name)

    for name, pipe in candidates:
        f1m, acc, _ = eval_one(name, pipe, X_train, y_train, X_val, y_val)
        if best is None or f1m > best[0]:
            best = (f1m, acc, name)

    print("\nBEST:", best[2], "macro_f1=", round(best[0], 4), "acc=", round(best[1], 4))

if __name__ == "__main__":
    main()
