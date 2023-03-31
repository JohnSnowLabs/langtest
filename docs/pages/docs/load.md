---
layout: docs
seotitle: Load | NLP Docs | John Snow Labs
title: Load
permalink: /docs/pages/docs/load
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
 
```python
# load saved configurations and test data
harness = h.load("saved_nlptest_folder", model="ner_dl_bert", task='ner', hub="johnsnowlabs")
```

Harness will load the saved nlptest configurations and test data. Now you can easily run the test cases with any new model
(ner_dl_bert, in our case). In order to run the test cases we can just use `harness.run()`.


</div></div>