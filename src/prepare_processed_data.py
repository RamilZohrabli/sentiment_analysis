from pathlib import Path
import pandas as pd
from preprocess import clean_text

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_clean.csv": RAW_DIR / "training.csv",
    "val_clean.csv":   RAW_DIR / "validation.csv",
}

def main():
    for out_name, in_path in FILES.items():
        df = pd.read_csv(in_path)
        df["text"] = df["text"].apply(clean_text)
        df.to_csv(OUT_DIR / out_name, index=False)
        print(f"Saved: {OUT_DIR / out_name} | shape={df.shape}")

if __name__ == "__main__":
    main()
