---
layout: docs
header: true
seotitle: NLP Tutorials | Stereotype | John Snow Labs
title: Stereotype
key: LLM_testing_Notebooks
permalink: /docs/pages/tutorials/LLM_testing_Notebooks/stereotype
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Stereotype notebook, we're evaluating `text-davinci-003` model on Stereotype Test, the primary goal of stereotype tests is to evaluate how well models perform when confronted with common gender stereotypes, occupational stereotypes, or other prevailing biases. In these assessments, models are scrutinized for their propensity to perpetuate or challenge stereotypical associations, shedding light on their capacity to navigate and counteract biases in their predictions.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |Dataset Used| Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Stereotype**                          | 	OpenAI/AI21                    | Question-Answering                               | `Wino-test`   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Wino_Bias_LLM.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used


```yml 
tests:
  defaults:
    min_pass_rate: 1.0
  
  stereotype:
    wino-bias:
      min_pass_rate: 0.70
```


<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`wino-bias`**: This test evaluates gender-based occupational stereotypes in LLM models, utilizing the Wino-bias dataset and methodology to assess gender bias in coreference resolution without relying on conventional stereotypes.


</div></div>

