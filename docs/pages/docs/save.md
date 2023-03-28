---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Saving Testcases
permalink: /docs/pages/docs/save
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Harness class provides `save()` and `load()` pipeline for the testcases. You can easily save and share generated 
testcases to run with different models. 
      
```python
#   access testcases
harness.testcases()

#   save testcases   
harness.save("path/to/nlptest_folder")
```

Harness will save generated testcasses, nlptest configurations and test data. `nlptest` saved folder can be used to test
 different NLP pipelines later using `harness.load()` method.
 
see [Loading testcases](/docs/pages/docs/load)  
```

</div></div>