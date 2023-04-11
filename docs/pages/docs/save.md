---
layout: docs
seotitle: Save | NLP Docs | John Snow Labs
title: Save/Load Workflow
permalink: /docs/pages/docs/save
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Save

Harness class provides `save()` and `load()` pipeline for the loading and saving the test configuration, generated test cases and the test data so that it can be reused later.
      
```python
# save testcases, test configuration, test data  
h.save("path/to/saved_nlptest_folder")
```

Harness will save generated test cases, nlptest configurations and test data. Later the saved folder can be used to test
 different NLP pipelines using `h.load()` method.

</div></div>