---
layout: docs
header: true
seotitle: String Distance Metrics | LangTest | John Snow Labs
title: String Distance Metrics
key: misc-examples
permalink: /docs/pages/misc/string_distance
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
We provides a collection of string distance metrics designed to quantify the similarity or dissimilarity between two strings. These metrics are useful in various applications where string comparison is needed. The available string distance metrics include:

{:.table2}
| Metric Name       | Description                       |
| ----------------- | --------------------------------- |
| jaro              | Measures the similarity between two strings based on the number of matching characters and transpositions. |
| jaro_winkler      | An extension of the Jaro metric that gives additional weight to common prefixes. |
| hamming           | Measure the difference between two equal-length sequences of symbols and is defined as the number of positions at which the corresponding symbols are different. |
| levenshtein       | Calculates the minimum number of single-character edits (insertions, deletions, substitutions) required to transform one string into another. |
| damerau_levenshtein | Similar to Levenshtein distance but allows transpositions as a valid edit operation. |
| Indel             | Focuses on the number of insertions and deletions required to match two strings. |

**Note:** returned scores are distances, meaning lower values are typically considered "better" and indicate greater similarity between the strings. The distances calculated are normalized to a range between 0.0 (indicating a perfect match) and 1.0 (indicating no similarity).

</div></div><div class="h3-box" markdown="1">

### Configuration Structure

To configure string distance metrics, you can use a YAML configuration file. The configuration structure includes:

- `model_parameters` specifying model-related parameters.
- `evaluation` setting the evaluation `metric`, `distance`, and `threshold`.
- `tests` defining different test scenarios and their `min_pass_rate`.

Here's an example of the configuration structure:

```yaml
model_parameters:
  temperature: 0.2
  max_tokens: 64

evaluation:
  metric: string_distance
  distance: jaro
  threshold: 0.1

tests:
  defaults:
    min_pass_rate: 1.0

  robustness:
    add_typo:
      min_pass_rate: 0.70
    lowercase:
      min_pass_rate: 0.70
```