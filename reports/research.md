# Sentiment / Emotion Analysis – Research Notes

## What is emotion classification?
Emotion classification is a supervised NLP task where a model maps a text to one of predefined emotions.
In this task we predict 6 classes: sadness, joy, love, anger, fear, surprise.

## Common approaches
### 1) Classical ML (strong baseline)
- Convert text into numeric features using Bag-of-Words or TF-IDF.
- Train a linear classifier (Logistic Regression / Linear SVM).
- Pros: fast, works well on small-medium datasets, easy to interpret.
- Cons: limited understanding of context/semantics.

### 2) Deep Learning
- RNN/LSTM/GRU: learns sequential patterns, but slower to train and struggles with long-range context.
- Transformers (BERT/DistilBERT): use self-attention to capture context and semantics.
  Fine-tuning a pre-trained transformer often improves performance.

## Why compare Baseline vs Transformer?
A professional workflow usually includes:
- Baseline model (quick, interpretable) to set a reference performance.
- Stronger model (Transformer) to improve accuracy and macro F1, especially for minority classes.

## Evaluation metrics
- Accuracy: overall correctness.
- Macro F1: average F1 across classes (important for imbalanced datasets like this one).
