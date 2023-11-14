---
layout: docs
header: true
seotitle: NLP Tutorials | John Snow Labs
title: Stereoset Notebook
key: test_specific
permalink: /docs/pages/tutorials/test_specific_notebooks/stereoset
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

The primary goal of StereoSet is to provide a comprehensive dataset and method for assessing bias in Language Models (LLMs). Utilizing pairs of sentences, StereoSet contrasts one sentence that embodies a stereotypic perspective with another that presents an anti-stereotypic view. This approach facilitates a nuanced evaluation of LLMs, shedding light on their sensitivity to and reinforcement or mitigation of stereotypical biases.


## Open in Collab

{:.table2}
| Test Type               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Stereoset** | Huggingface                    | Question-Answering                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/task-specific-notebooks/StereoSet_Notebook.ipynb)                                |



## Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0

  stereoset:
    intrasentence:
      min_pass_rate: 0.70
      diff_threshold: 0.1
    intersentence:
      min_pass_rate: 0.70
      diff_threshold: 0.1
```