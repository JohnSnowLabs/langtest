---
layout: docs
header: true
seotitle: NLP Tutorials | John Snow Labs
title: Bias tests with transformers
key: tutorials
permalink: /docs/pages/tutorials/bias_tests
sidebar:
    nav: tutorials
aside:
    toc: true
show_edit_on_github: true
nav_key: tutorials
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
To install the **johnsnowlabs Python library** and all of John Snow Labs open **source libraries**, just run

```shell 
pip install johnsnowlabs
```

To quickly test the installation, you can run in your **Shell**:

```shell
python -c "from johnsnowlabs import nlp;print(nlp.load('emotion').predict('Wow that easy!'))"
```
or in **Python**:
```python
from  johnsnowlabs import nlp
nlp.load('emotion').predict('Wow that easy!')
```
</div></div>