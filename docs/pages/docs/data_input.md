---
layout: docs
header: true
seotitle: Data Inputs | NLP Docs | John Snow Labs
title: Data Inputs
key: docs-examples
permalink: /docs/pages/docs/data_input
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Supported data inputs are **`CoNLL`** and **`CSV`** formats. CoNLL dataset can only be loaded for `ner`. For the `text-classification`, both formats are supported provided the column names are from a list of supported column names.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL and CSV|
|**text-classification**     |CSV

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

<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 20px;
    padding-bottom: 20px;
  }

  #NER {
    color: #1E77B7;
    font-size: 16px;
  }

</div></div>