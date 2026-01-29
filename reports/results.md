# Results & Interpretation

## Dataset summary
- Train: 16,000
- Validation: 2,000
- Test: 2,000
Classes are imbalanced (e.g., surprise is the smallest class), therefore macro F1 is important.

## Validation results
- Baseline (TF-IDF + LinearSVC): Accuracy = 0.9030, Macro F1 = 0.8735
- Transformer (DistilBERT, 1 epoch): Accuracy = 0.9255, Macro F1 = 0.8987

## Test results (final)
- Baseline (TF-IDF + LinearSVC): Accuracy = 0.8870, Macro F1 = 0.8395
- Transformer (DistilBERT, 1 epoch): Accuracy = 0.9235, Macro F1 = 0.8855

## Error analysis (high-level)
From confusion matrices:
- joy vs love: can overlap because both include positive wording.
- fear vs surprise: sometimes confused in short or ambiguous sentences.
- sadness performs strongest and is less ambiguous.

## Takeaways
- Transformer significantly improves generalization on the test set.
- Remaining errors mostly come from semantic overlap between emotions rather than preprocessing issues.
