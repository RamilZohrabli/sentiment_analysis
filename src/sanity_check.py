from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
TRAIN_PATH = RAW_DIR / "training.csv"
VAL_PATH = RAW_DIR / "validation.csv"
TEST_PATH = RAW_DIR / "test.csv"

# Label mapping based on emotions.txt
LABEL2EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_csv(path)

def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:\n", df.isna().sum())

    print("\nSample rows:")
    print(df.head(3))

    if "label" in df.columns:
        print("\nLabel distribution:")
        counts = df["label"].value_counts().sort_index()
        print(counts)

        #mapping
        print("\nLabel distribution (with emotion names):")
        for k, v in counts.items():
            print(f"{k} ({LABEL2EMOTION.get(k, 'UNKNOWN')}): {v}")

    if "text" in df.columns:
        lengths = df["text"].astype(str).str.split().str.len()
        print("\nText length (words) stats:")
        print(lengths.describe())

def main() -> None:
    train_df = load_csv(TRAIN_PATH)
    val_df = load_csv(VAL_PATH)
    test_df = load_csv(TEST_PATH)

    summarize(train_df, "Train")
    summarize(val_df, "Validation")
    summarize(test_df, "TEST")

if __name__ == "__main__":
    main()
