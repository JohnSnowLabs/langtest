---
layout: docs
seotitle: Load | LangTest | John Snow Labs
title: Load
permalink: /docs/pages/docs/load
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
 

## Load Test Cases
The Harness is able to load the saved test cases, test configuration and test data. 

```python
# Load saved test cases, test configuration, test data  
harness = Harness.load(
    save_dir="path/to/saved_test_folder",
    model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
    task='question-answering', 
    load_testcases=True
)
```

Once the harness is loaded, the test cases can then be run with any new model by calling `harness.run()`.

</div><div class="h3-box" markdown="1">

## Load Model response 

Load model responses along with the test harness configuration.

```python
harness = Harness.load(
    save_dir="saved_model_reponse",
    model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
    task="question-answering",
    load_model_response=True
)
```

After loading the harness, you can re-evaluate the model responses using alternative metrics