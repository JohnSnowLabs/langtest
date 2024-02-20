---
layout: docs
seotitle: LangTestCallback | LangTest | John Snow Labs
title: LangTestCallback
permalink: /docs/pages/docs/hf-callback
key: docs-callback
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

LangTest also has a callback class that can be used in training to evaluate the model after each epoch or at the end of training. This callback class is called `LangTestCallback` and is imported from `langtest.callback`.

```python
from langtest.callback import LangTestCallback
my_callback = LangTestCallback(task, config, data)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[my_callback]
)
```
</div><div class="h3-box" markdown="1">

LangTestCallback takes the following parameters:

{:.table2}
| Parameter          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **task**           | Task for which the model is to be evaluated (text-classification or ner)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **data**           | The data to be used for evaluation. A dictionary providing flexibility and options for data sources. It should include the following keys: <br/> -  **data_source** (mandatory): The source of the data. <br/> -  **subset** (optional): The subset of the data. <br/> - **feature_column** (optional): The column containing the features. <br/> - **target_column** (optional): The column containing the target labels. <br/> - **split** (optional): The data split to be used. <br/> - **source** (optional): Set to 'huggingface' when loading Hugging Face dataset. |
| **config**         | Configuration for the tests to be performed, specified in the form of a YAML file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **print_reports**  | A bool value that specifies if the reports should be printed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **save_reports**   | A bool value that specifies if the reports should be saved. If `True`, all generated reports will be saved under `reports/reportXXX.md`                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **run_each_epoch** | A bool value that specifies if the tests should be run after each epoch or the at the end of training                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |


</div>