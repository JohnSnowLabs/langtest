---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Loading Testcases
permalink: /docs/pages/docs/load
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Harness class provides `save()` and `load()` pipeline for the testcases. You can easily load and run generated 
testcases to from saved nlptest folder.
      
```python
from nlptest import Harness

#   load testcases
harness = Harness.load("saved_nlptest_folder", task='ner', model="ner_dl_bert", hub="johnsnowlabs")
```

Harness will load saved testcasses, nlptest configurations and test data. Now you can easily run saved test casses with
`ner_dl_bert` model that you passed. In order to `run()` testcasses see [Running testcases](/docs/pages/docs/run)

</div></div>