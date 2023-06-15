---
layout: docs
seotitle: Test Harness | NLP Test | John Snow Labs
title: Test Harness
permalink: /docs/pages/docs/harness
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Harness Class

The Harness class is a testing class for NLP models. It evaluates the performance of a given NLP model on a given task, dataset and test configuration. Users are able to **generate test cases**, **save and re-use them**, **create reports** and **augment** datasets based on test results.

```python
# Import Harness from the nlptest library
from nlptest import Harness
```

Here is a list of the different parameters that can be passed to the `Harness` class:

</div><div class="h3-box" markdown="1">

### Parameters
 

{:.table2}
| Parameter   | Description |  
| - | - | 
|**task**     |Task for which the model is to be evaluated ('text-classification', 'question-answering', 'ner')|
|**model**    |Pretrained pipeline or model from the corresponding hub, or path to a saved model from the corresponding hub, or PipelineModel object or a dictionary containing the names of the models you want to compare, each paired with its respective hub - see [Model Input](https://nlptest.org/docs/pages/docs/model_input) for more details
|**hub**      |Hub (library) to use in back-end for loading model from public models hub or from path|
|**data**     |Path to the data to be used for evaluation. Should be `.csv` or a dictionary containing the name, subset, split, feature_column and target_column for loading the HF dataset for text classification, or `.conll` or `.txt` file in CoNLL format for NER - see [Data Input](https://nlptest.org/docs/pages/docs/data_input) for more details
|**config**   |Path to the YAML file with configuration of tests to be performed

</div></div>