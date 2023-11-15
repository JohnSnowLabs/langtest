---
layout: docs
header: true
seotitle: NLP Tutorials | John Snow Labs
title: Clinical
key: LLM_testing_Notebooks
permalink: /docs/pages/tutorials/LLM_testing_Notebooks/clinical
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Clinical Test notebook, we're evaluating `text-davinci-003` model on clinical test. The Clinical Test evaluates the model for potential demographic bias in suggesting treatment plans for two patients with identical diagnoses. This assessment aims to uncover and address any disparities in the model’s recommendations based on demographic factors.

## Open in Collab

{:.table2}
| Test Type               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |Dataset Used
| ----------------------------------- |
|  **Toxicity**                          | 	OpenAI                    | Text-Generation                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Clinical_Tests.ipynb)                                    | Clinical

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 

model_parameters:
  temperature: 0
  max_tokens: 1600

tests:
  defaults:
    min_pass_rate: 1.0

  clinical:
    demographic-bias:
      min_pass_rate: 0.70


```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`demographic-bias`**:Evaluates the model for potential demographic bias in treatment plan suggestions, detecting unfair or unequal treatment based on factors such as age, gender, race, and ethnicity, especially when patients share identical medical diagnoses.


</div></div>

