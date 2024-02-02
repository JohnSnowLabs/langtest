---
layout: docs
header: true
seotitle: Embedding Distance Metrics | LangTest | John Snow Labs
title: Embedding Distance Metrics
key: misc-examples
permalink: /docs/pages/misc/embedding_distance
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
We offers a range of embedding models from different hubs, with two default models preconfigured:

{:.table2}
| **Hub** | **Default Model**                |
| --------------------- | -----------------------          |
| OpenAI           | text-embedding-ada-002        |
| HuggingFace      | sentence-transformers/all-mpnet-base-v2 |


> Users can specify the desired embedding model and hub to generate embeddings for the *expected_result* and *actual_result*. These embeddings can then be compared using various distance metrics defined in the configuration.


When comparing embeddings, it's crucial to use the appropriate distance metric. The library supports several distance metrics for this purpose:

{:.table2}
| Metric Name       | Description                       |
| ----------------- | --------------------------------- |
| Cosine similarity | Measures the cosine of the angle between two vectors. |
| Euclidean distance | Calculates the straight-line distance between two points in space. |
| Manhattan distance | Computes the sum of the absolute differences between corresponding elements of two vectors. |
| Chebyshev distance | Determines the maximum absolute difference between elements in two vectors. |
| Hamming distance  | Measure the difference between two equal-length sequences of symbols and is defined as the number of positions at which the corresponding symbols are different. |

</div></div><div class="h3-box" markdown="1">

### Configuration Structure

To configure your embedding models and evaluation metrics, you can use a YAML configuration file. The configuration structure includes:

- `model_parameters` specifying model-related parameters.
- `evaluation` setting the evaluation `metric`, `distance`, and `threshold`.
- `embeddings` allowing you to choose the embedding `model` and `hub`.
- `tests` defining different test scenarios and their `min_pass_rate`.

Here's an example of the configuration structure:

```yaml
model_parameters:
  temperature: 0.2
  max_tokens: 64

evaluation:
  metric: embedding_distance
  distance: cosine
  threshold: 0.8

embeddings:
  model: text-embedding-ada-002
  hub: openai

tests:
  defaults:
    min_pass_rate: 1.0

  robustness:
    add_typo:
      min_pass_rate: 0.70
    lowercase:
      min_pass_rate: 0.70
```