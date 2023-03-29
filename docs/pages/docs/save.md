---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Save/Load Workflow
permalink: /docs/pages/docs/save
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

<div class="heading" id="saving">Save</div>

Harness class provides `save()` and `load()` pipeline for the loading and saving the test configuration, generated testcases and the test data so that it can be reused later.
      
```python
#  import harness
from nlptest import Harness

#   save testcases, test configuration, test data  
harness.save("path/to/saved_nlptest_folder")
```

Harness will save generated test cases, nlptest configurations and test data. Later the saved folder can be used to test
 different NLP pipelines using `harness.load()` method.

<div class="heading" id="loading">Load</div>
 
```python
#   import harness
from nlptest import Harness

#   load saved testcases, config and test data
harness = Harness.load("saved_nlptest_folder", task='ner', model="ner_dl_bert", hub="johnsnowlabs")
```

Harness will load saved testcases, nlptest configurations and test data. Now you can easily run saved test cases with any model
(`ner_dl_bert`) in our case. In order to run the test cases we can just use `harness.run()`.

 <div class="heading" id="saving-testcase">Saving Test Cases</div>

 In order to save the generated test cases, we can make use of the **`.save_testcases()`** method. It saves the generated test cases in the form of a pickle file which can be then loaded and inspected.

```python
#  import harness
from nlptest import Harness

#  save testcases
Harness.save_testcases("save_path")
```
It saves the testcases as a pickle file to the specified location (save_path). 


 <div class="heading" id="loading-testcase">Loading Test Cases</div>

 In order to load the saved generated test cases, we can make use of the **`.load_testcases()`** method. It loads the generated test cases from the saved pickle file.

```python
#  import harness
from nlptest import Harness

#  load saved testcases
Harness.load_testcases("save_path")
```
It loads the saved test cases from the specified location (save_path). 


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

  #saving-testcase {
    color: #1E77B7;
  }
  
  #loading-testcase {
    color: #1E77B7;
  }
  
  

</style>

</div></div>