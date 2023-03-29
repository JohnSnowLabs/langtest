---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Quick Start
permalink: /docs/pages/docs/install
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

**nlptest** is an open-source Python library designed to help developers deliver safe and effective Natural Language Processing (NLP) models.
You can install **nlptest** using pip or conda.

<div class="heading" id="installation"> Installation </div>

```python 
# Using PyPI
pip install nlptest

# Using Conda
conda install nlptest
```

<div class="heading" id="one-liners"> One Liners </div>

With just one line of code, it can generate and run over 50 different test types to assess the quality of NLP models in terms of accuracy, bias, robustness, representation, and fairness.
You can test any **Text Classification** and **Named Entity Recognition** model using ``Harness``.

```python
from nlptest import Harness
h = Harness(task='ner', model='ner_dl_bert', hub='johnsnowlabs')

# Generate test cases, run them and view a report
h.generate().run().report()
```

Whether you are using **Spark NLP**, **Hugging Face Transformers**, or **spaCy** models, ``Harness`` has got you covered.
```python
from nlptest import Harness
h = Harness(task='text-classification', model='mrm8488/distilroberta-finetuned-tweets-hate-speech', hub='huggingface')

# Generate test cases, run them and view a report
h.generate().run().report()
```
<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 20px;
    padding-bottom: 20px;
  }

  #installation {
    color: #1E77B7;
  }
  
  #one-liners {
    color: #1E77B7;
  }
  

</div></div>