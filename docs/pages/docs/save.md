---
layout: docs
seotitle: Save | LangTest | John Snow Labs
title: Save
permalink: /docs/pages/docs/save
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Saving Test Cases

The Harness class provides `save()` and `load()` methods to save and load test configurations, generated test cases and the test data so that it can be reused later.
      
```python
# Save test cases, test configuration, test data  
h.save(save_dir="path/to/saved_test_folder")
```
</div><div class="h3-box" markdown="1">

## Saving Model Responses

After executing the `.run()` method, you can save model responses for re-evaluation and analysis.

```python 
h.save(
    save_dir="model_response", 
    include_generated_results=True
)
```