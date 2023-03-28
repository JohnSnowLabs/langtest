---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Harness and its Parameters
permalink: /docs/pages/docs/harness
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The Harness class is a testing class for Natural Language Processing (NLP) models. It evaluates the performance of a given NLP model on a given task using test data and generates a report with test results. Harness can be imported from the nlptest library in the following way.

```python
#Import Harness from the nlptest library
from nlptest import Harness
```

It imports the Harness class from within the module, that is designed to provide a blueprint or framework for conducting NLP testing, and that instances of the Harness class can be customized or configured for different testing scenarios or environments.

Here is a list of the different parameters that can be passed to the `Harness` function:

| Parameter  | Description |  |
| - | - | - |
|**task**      |Task for which the model is to be evaluated (text-classification or ner)|
|**model** |PipelineModel or path to a saved model or pretrained pipeline/model from hub.
|**data**     |Path to the data that is to be used for evaluation. Can be .csv or .conll file in <br/> the CoNLL format 
|**config**      |Configuration for the tests to be performed, specified in form of a YAML file.
|**hub**      |model hub to load from the path. Required if model param is passed as path.|




</div></div>