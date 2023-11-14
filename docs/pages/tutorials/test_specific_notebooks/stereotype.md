---
layout: docs
header: true
seotitle: NLP Tutorials | John Snow Labs
title: Stereotype Notebooks
key: test_specific
permalink: /docs/pages/tutorials/test_specific_notebooks/stereotype
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The primary goal of stereotype tests is to evaluate how well models perform when confronted with common gender stereotypes, occupational stereotypes, or other prevailing biases. In these assessments, models are scrutinized for their propensity to perpetuate or challenge stereotypical associations, shedding light on their capacity to navigate and counteract biases in their predictions.

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Common Stereotypes Notebook

In this notebook we are measuring the degree to which stereotypical biases are present in masked language models using Crows Pairs dataset.

### Open in Collab

{:.table2}
| Test Type               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Common-Stereotypes** | Hugging Face                      | Fill-Mask                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/task-specific-notebooks/Crows_Pairs_Notebook.ipynb)                         |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0
  crows-pairs:
    common-stereotypes:
      min_pass_rate: 0.7
      diff_threshold: 0.1
      filter_threshold: 0.15
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Gender Occupational Stereotype Notebook

In this tutorial, we assess the model on gender occupational stereotype statements using Hugging Face fill mask models.

### Open in Collab

{:.table2}
| Test Type               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Gender-Occupational-Stereotype**| Hugging Face                      | Fill-Mask                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/task-specific-notebooks/Wino_Bias.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0
  wino-bias:
    gender-occupational-stereotype:
      min_pass_rate: 0.7
      diff_threshold: 0.03

```

</div></div>