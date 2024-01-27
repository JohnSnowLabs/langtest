---
layout: docs
seotitle: Test Harness | LangTest | John Snow Labs
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
# Import Harness from the LangTest library
from langtest import Harness
```

Here is a list of the different parameters that can be passed to the `Harness` class:

</div><div class="h3-box" markdown="1">

### Parameters
 

{:.table2}
| Parameter   | Description |  
| - | - | 
|[**task**](/docs/pages/docs/task)     |Task for which the model is to be evaluated ('text-classification', 'question-answering', 'ner')|
| [**model**](/docs/pages/docs/model)     | Specifies the model(s) to be evaluated. This parameter can be provided as either a dictionary or a list of dictionaries. Each dictionary should contain the following keys: <BR>• model (mandatory): 	PipelineModel or path to a saved model or pretrained pipeline/model from hub.<BR>• hub (mandatory): Hub (library) to use in back-end for loading model from public models hub or from path|
| [**data**](/docs/pages/docs/data)      | The data to be used for evaluation. A dictionary providing flexibility and options for data sources. It should include the following keys: <BR>• data_source (mandatory): The source of the data.<BR>• subset (optional): The subset of the data.<BR>• feature_column (optional): The column containing the features.<BR>• target_column (optional): The column containing the target labels.<BR>• split (optional): The data split to be used.<BR>• source (optional): Set to 'huggingface' when loading Hugging Face dataset. |
|[**config**](/docs/pages/docs/config)   |Path to the YAML file with configuration of tests to be performed

</div></div>