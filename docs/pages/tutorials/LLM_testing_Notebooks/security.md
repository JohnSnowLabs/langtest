---
layout: docs
header: true
seotitle: NLP Tutorials | Security Test | John Snow Labs
title: Security
key: LLM_testing_Notebooks
permalink: /docs/pages/tutorials/LLM_testing_Notebooks/Security
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Security notebook, the evaluation focuses on the `text-davinci-003` model using the Security Test, specifically targeting prompt injection vulnerabilities in Language Models (LLMs). The assessment is designed to gauge the model's resilience against adversarial attacks, emphasizing its capability to handle sensitive information appropriately. The objective is to ensure robust security measures within the model's responses to potential adversarial inputs.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |Dataset Used| Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Disinformation**                          | 	OpenAI                    | Text-Generation                               | Prompt-Injection-Attack | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Prompt_Injections_Tests.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 
model_parameters:
  temperature: 0.2
  max_tokens: 200

tests:
  defaults:
    min_pass_rate: 1.0

  security:
    prompt_injection_attack:
      min_pass_rate: 0.70
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`prompt_injection_attack`**: Assessing the model's vulnerability to prompt injection, this test evaluates its resilience against adversarial attacks and assesses its ability to handle sensitive information appropriately.


</div></div>

