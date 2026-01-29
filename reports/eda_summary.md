## EDA Summary

- Dataset split:
  - Train: 16,000 rows
  - Validation: 2,000 rows
  - Test: 2,000 rows

- Label distribution (train):
  - sadness (0): 4666
  - joy (1): 5362
  - love (2): 1304
  - anger (3): 2159
  - fear (4): 1937
  - surprise (5): 572

- Observation:
  - Classes are imbalanced: 'joy' and 'sadness' dominate, while 'surprise' is the smallest class.
  - Because of this, **macro F1-score** will be used in addition to accuracy to better reflect performance across all emotions.

- Text length (train, words):
  - mean: 19.17, median: 17
  - 75% of texts are <= 25 words
  - max length: 66 words
  - This indicates that most samples are short, and (later) a max sequence length of 128 tokens will be sufficient for transformer-based models.
