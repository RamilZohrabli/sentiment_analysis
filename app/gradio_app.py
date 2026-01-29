from pathlib import Path
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION2LABEL = {v: k for k, v in LABEL2EMOTION.items()}

MODEL_DIR = Path("models/transformer/best_model")
FALLBACK_MODEL = "distilbert-base-uncased"

MAX_LEN = 128

def load_model():
    if MODEL_DIR.exists():
        model_path = str(MODEL_DIR)
    else:
        model_path = FALLBACK_MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=6,
        id2label=LABEL2EMOTION,
        label2id=EMOTION2LABEL,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device, model_path

tokenizer, model, device, loaded_from = load_model()

def predict(text: str):
    text = (text or "").strip()
    if not text:
        return "Please enter a sentence.", {}

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    pred_id = int(probs.argmax())
    pred_label = LABEL2EMOTION[pred_id]
    prob_dict = {LABEL2EMOTION[i]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_dict

examples = [
    ["I feel empty and hopeless today."],                 # sadness
    ["I just got accepted and I'm so happy!"],            # joy
    ["I adore you and feel grateful to have you."],       # love
    ["I'm furious about what happened."],                 # anger
    ["I'm scared something bad will happen."],            # fear
    ["Wow, I didn't expect that at all!"],                # surprise
]

with gr.Blocks(title="Emotion Classifier") as demo:
    gr.Markdown(
        f"""
# Emotion Classifier (6 classes)

**Loaded model from:** `{loaded_from}`  
Type a sentence and get the predicted emotion + class probabilities.
"""
    )

    inp = gr.Textbox(
        label="Input sentence",
        placeholder="Type something like: I feel amazing today!",
        lines=3,
    )

    btn = gr.Button("Predict")
    out_pred = gr.Textbox(label="Predicted emotion")
    out_probs = gr.Label(label="Probabilities", num_top_classes=6)

    btn.click(fn=predict, inputs=inp, outputs=[out_pred, out_probs])

    gr.Markdown("## Quick examples")
    gr.Examples(examples=examples, inputs=inp)

demo.launch()
