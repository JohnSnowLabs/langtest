---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Test Harness
permalink: /docs/pages/docs/harness
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

<div class="heading" id="harness">Harness Explained</div>

The Harness class is a testing class for Natural Language Processing (NLP) models. It evaluates the performance of a given NLP model on a given task using test data and generates a report with test results. Harness can be imported from the nlptest library in the following way.

```python
#Import Harness from the nlptest library
from nlptest import Harness

```

It imports the Harness class from within the module, that is designed to provide a blueprint or framework for conducting NLP testing, and that instances of the Harness class can be customized or configured for different testing scenarios or environments.

Here is a list of the different parameters that can be passed to the `Harness` function:

<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 20px;
    padding-bottom: 20px;
  }

  #parameters {
    color: #1E77B7;
  }

  #NER {
    color: #1E77B7;
    font-size: 16px;
  }

  
  #harness {
    color: #1E77B7;
  }

  #data_input {
    color: #1E77B7;
  }
  

</style>

<div class="heading" id="parameters">Parameters</div>
 

{:.table2}
| Parameter  | Description |  
| - | - | 
|**task**     |Task for which the model is to be evaluated (text-classification or ner)|
|**model**     |PipelineModel or path to a saved model or pretrained pipeline/model from hub.
|**data**       |Path to the data that is to be used for evaluation. Can be .csv or .conll file in the CoNLL format 
|**config**     |Configuration for the tests to be performed, specified in form of a YAML file.
|**hub**       |model hub to load from the path. Required if model param is passed as path.|



<div class="heading" id="data_input">Data Input</div>
Supported data inputs are **`CoNLL`** and **`csv`** formats. CoNLL dataset can only be loaded for `ner`. For the `text-classification`, both formats are supported provided the column names are from a list of supported column names.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL and Csv|
|**text-classification**     |Csv

<div class="heading" id="NER"> For NER  </div>

```python
#Import Harness from the nlptest library
from nlptest import Harness
harness = Harness(
            task='ner',
            model='en_core_web_sm',
            config= 'sample_config.yml',
            hub = "spacy",
            data= 'sample.conll/sample.csv'  
         
        )
```

<div class="heading" id="NER"> For Text-Classification  </div>

```python
#Import Harness from the nlptest library
from nlptest import Harness
harness = Harness(
            task='text-classification',
            model='mrm8488/distilroberta-finetuned-tweets-hate-speech',
            config= 'sample_config.yml',
            hub = "huggingface",
            data= 'sample.csv'  #CoNLL format not supported
         
        )

```


</div></div>