from pathlib import Path
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


LABEL2EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTIONS_ORDER = [LABEL2EMOTION[i] for i in sorted(LABEL2EMOTION.keys())]

MODEL_NAME = "distilbert-base-uncased"

TRAIN_PATH = Path("data/processed/train_clean.csv")
VAL_PATH = Path("data/processed/val_clean.csv")

OUT_DIR = Path("models/transformer")
BEST_DIR = OUT_DIR / "best_model"
REPORT_DIR = Path("reports/metrics")

OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class TextEmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextEmotionDataset(train_df, tokenizer, max_length=128)
    val_dataset = TextEmotionDataset(val_df, tokenizer, max_length=128)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        id2label=LABEL2EMOTION,
        label2id={v: k for k, v in LABEL2EMOTION.items()},
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        #CPU friendly
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,

        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none",

        seed=42,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train+evaluate
    trainer.train()
    eval_res = trainer.evaluate()
    print("\nFINAL VALIDATION:", eval_res)

    # Save
    trainer.save_model(str(BEST_DIR))
    tokenizer.save_pretrained(str(BEST_DIR))
    print(f"Saved model to: {BEST_DIR}")

    #detailed report + confusion matrix
    pred_out = trainer.predict(val_dataset)
    logits = pred_out.predictions
    y_true = pred_out.label_ids
    y_pred = np.argmax(logits, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=EMOTIONS_ORDER,
        digits=4
    )
    (REPORT_DIR / "transformer_classification_report.txt").write_text(report, encoding="utf-8")
    print("\nCLASSIFICATION REPORT (VAL):\n", report)

    cm = confusion_matrix(y_true, y_pred, labels=sorted(LABEL2EMOTION.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS_ORDER)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Transformer Confusion Matrix (Validation)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "transformer_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"Saved confusion matrix to: {REPORT_DIR / 'transformer_confusion_matrix.png'}")
    print("DONE")


if __name__ == "__main__":
    main()
