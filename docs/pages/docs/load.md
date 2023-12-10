---
layout: docs
seotitle: Loading Tests | LangTest | John Snow Labs
title: Loading Tests
permalink: /docs/pages/docs/load
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
 
The Harness is able to load the saved test cases, test configuration and test data. 

```python
# Load saved test cases, test configuration, test data  
harness = Harness.load(save_dir="path/to/saved_test_folder",
model={"model": "ner_dl_bert", "hub":"johnsnowlabs"},
task="ner",
load_testcases=True)
```

Once the harness is loaded, the test cases can then be run with any new model by calling `harness.run()`.

</div></div>