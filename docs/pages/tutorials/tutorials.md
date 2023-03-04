---
layout: docs
header: true
seotitle: NLP Tutorials | John Snow Labs
title: Tutorials
key: docs-tutorials
permalink: /docs/pages/tutorials/tutorials
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
show_edit_on_github: true
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
To install the **johnsnowlabs Python library** and all of John Snow Labs open **source libraries**, just run

```shell 
pip install johnsnowlabs
```

## Test h2


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