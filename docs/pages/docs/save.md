---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Saving/Loading Workflow
permalink: /docs/pages/docs/save
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

<div class="heading" id="saving">Saving Test Cases</div>

Harness class provides `save()` and `load()` pipeline for the loading and saving the testcases. 
      
```python
#  import harness
from nlptest import Harness

#   save testcases   
harness.save("path/to/nlptest_folder")
```

Harness will save generated test cases, nlptest configurations and test data. Later `nlptest` saved folder can be used to test
 different NLP pipelines using `harness.load()` method.

 <div class="heading" id="loading">Loading Test Cases</div>
 
```python
#   import harness
from nlptest import Harness

#   load testcases
harness = Harness.load("saved_nlptest_folder", task='ner', model="ner_dl_bert", hub="johnsnowlabs")
```

Harness will load saved testcases, nlptest configurations and test data. Now you can easily run saved test cases with
`ner_dl_bert` model that you passed. In order to run the test cases use `harness.run()`.

<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 20px;
    padding-bottom: 20px;
  }

  #saving {
    color: #1E77B7;
  }
  
  #loading {
    color: #1E77B7;
  }
  

</style>

</div></div>