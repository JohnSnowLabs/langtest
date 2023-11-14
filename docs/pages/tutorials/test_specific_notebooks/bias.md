---
layout: docs
header: true
seotitle: NLP Tutorials | John Snow Labs
title: Bias Notebook
key: test_specific
permalink: /docs/pages/tutorials/test_specific_notebooks/bias
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In this Bias notebook, our main goal is checking if the ner.dl model has any biases. Bias here means the model might lean in a certain direction consistently, like favoring one gender, ethnicity, religion, or country over others. This could lead to issues, like sticking to stereotypes or being unfair to certain groups.

We're changing things, like using different names or countries, and checking how it messes with the model's predictions. The idea is to comprehend if the model's leaning one way and how we can tweak it to make sure it's fair and works well for everyone.

## Open in Collab

{:.table2}
| Test Type               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Bias**                          | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/test-specific-notebooks/Bias_Demo.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:     
  defaults:
    min_pass_rate: 0.65
  bias:
    replace_to_female_pronouns:
      min_pass_rate: 0.66
    replace_to_hindu_names:
      min_pass_rate: 0.60
```

</div></div>