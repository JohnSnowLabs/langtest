---
layout: docs
header: true
seotitle: NLP Tutorials | QA-Summarization Test | John Snow Labs
title: QA & Summarization Notebook
key: test_specific
permalink: /docs/pages/tutorials/LLM_testing_Notebooks/QA_Summarization
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the QA/Summarization notebook section, our primary focus is on the evaluation of OpenAI, Ai21, Cohere, Asure-OpenAi models tailored for Question Answering (QA) and Summarization tasks. This involves a meticulous testing process to gauge the models' efficiency in responding to questions and producing concise and informative summaries. In a noteworthy addition, we're introducing perturbations to the text during these tasks. By incorporating variations in the input, such as adding perturbations, we aim to observe how the model responses adapt to changes in the text. 

## Config Used

```yml
tests:
  defaults:
    min_pass_rate: 1.0

  robustness:
    add_typo:
      min_pass_rate: 0.70
    lowercase:
      min_pass_rate: 0.70
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## OpenAi Question-Answering & Summarization

In this notebook we are testing OpenAI `text-davinci-003` Model For Question Answering and Summarization task.


### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
| **OpenAI QA/Summarization** | OpenAI                            | Question-Answering/Summarization  | `BoolQ`, `NQ-Open`, `Xsum` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/OpenAI_QA_Summarization_Testing_Notebook.ipynb)               |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## AI21 Question-Answering & Summarization

In this notebook we are testing AI21 `j2-jumbo-instruct` Model For Question Answering and Summarization task.


### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
|  **Question-Answering & Summarization**   | AI21                              | Question-Answering/Summarization  | `BoolQ`, `NQ-Open`, `Xsum` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/AI21_QA_Summarization_Testing_Notebook.ipynb)    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Cohere Question-Answering & Summarization

In this notebook we are testing Cohere `command-xlarge-nightly` Model For Question Answering and Summarization task.


### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
|  **Question-Answering & Summarization**   | Cohere                            | Question-Answering/Summarization  | `BoolQ`, `NQ-Open`, `Xsum` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Cohere_QA_Summarization_Testing_Notebook.ipynb) |

0987   <div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Hugging Face Inference API Question-Answering & Summarization

In this notebook we are testing Hugging Face Inference API `google/flan-t5-small` Model For Question Answering and `google/pegasus-newsroom` for Summarization task.

### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
|  **Question-Answering & Summarization**           | Hugging Face Inference API        | Question-Answering/Summarization  | `BoolQ`, `NQ-Open`, `Xsum` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/HuggingFaceAPI_QA_Summarization_Testing_Notebook.ipynb)       |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Hugging Face Hub Question-Answering & Summarization

In this notebook we are testing Hugging Face Hub `facebook/opt-1.3b` Model For Question Answering and Summarization task.

### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
|  **Question-Answering & Summarization**   | Hugging Face Hub                  | Question-Answering/Summarization  | `BoolQ`, `NQ-Open`, `Xsum` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/HuggingFaceHub_QA_Summarization_Testing_Notebook.ipynb)       |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Azure-OpenAI Question-Answering & Summarization

In this notebook we are testing Azure-OpenAI `text-davinci-003` Model For Question Answering and Summarization task.


### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
|  **Question-Answering & Summarization**  | Azure-OpenAI                      | Question-Answering/Summarization  | `BoolQ`, `NQ-Open`, `Xsum` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Azure_OpenAI_QA_Summarization_Testing_Notebook.ipynb) |

</div></div>