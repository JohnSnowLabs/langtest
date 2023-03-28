---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Installation
permalink: /docs/pages/docs/install
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

**nlptest** is an open-source Python library designed to help developers deliver safe and effective Natural Language Processing (NLP) models.
You can install **nlptest** using pip or conda.

```python 
# Using PyPI
pip install nlptest

# Using Conda
conda install nlptest
```

With just one line of code, it can generate and run over 50 different test types to assess the quality of NLP models in terms of accuracy, bias, robustness, representation, and fairness.
You can test any **Text Classification** and **Named Entity Recognition** model using ``Harness``.

```python
from nlptest import Harness
h = Harness(task='ner', model='ner_dl_bert', hub='johnsnowlabs')

# Generate test cases, run them and view a report
h.generate().run().report()
```

Whether you are using **Spark NLP**, **Hugging Face Transformers**, or **spaCy** models, ``Harness`` has got you covered.
You can easily pass the test data and the trained NLP pipeline.
```python
from nlptest import Harness
h = Harness(task='text-classification', model='distilbert-base-uncased', hub='huggingface')

# Generate test cases, run them and view a report
h.generate().run().report()
```

</div></div>