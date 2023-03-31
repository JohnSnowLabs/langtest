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

Supported data inputs formats are **`CoNLL`** and **`CSV`**. CoNLL dataset can only be loaded for `ner`. For `text-classification`, both formats are supported provided the column names are from a list of supported column names.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL and CSV|
|**text-classification**     |CSV


<div class="heading" id="NER"> Sample CoNLL format </div>

```bash
LEICESTERSHIRE NNP B-NP B-ORG
TAKE NNP I-NP O
OVER IN B-PP O
AT NNP B-NP O
TOP NNP I-NP O
AFTER NNP I-NP O
INNINGS NNP I-NP O
VICTORY NN I-NP O
. . O O
```

<div class="heading" id="NER"> Sample CSV format </div>

A sample CSV data input looks like the following : 

{:.table2}
| text | label  |  
| - | - | 
|This is a pretty faithful adaptation of Masuji Ibuse's novel, "Black Rain." Like the book it is very moving and thought-provoking. The story revolves around a couple's attempts to see their niece successfully married. They are having trouble finding suitors because of a rumor that she suffers from radiation sickness, after walking through Hiroshima on the day of the bombing. Well filmed, well acted, moving, tragic, horrifying and funny.| 1 |

For `CSV` files, we support different variations of the column names. They are shown below :

<div class="heading" id="NER"> For Text-Classification </div>

{:.table2}
| Supported "text" column names | Supported "label" column names   |  
| - | - | 
| ['text', 'sentences', 'sentence', 'sample'] | ['label', 'labels ', 'class', 'classes'] |


<div class="heading" id="NER"> For NER </div>

{:.table2}
| Supported "text" column names | Supported "ner" column names | Supported "pos" column names | Supported "chunk" column names | 
| - | - | 
| ['text', 'sentences', 'sentence', 'sample'] |  ['label', 'labels ', 'class', 'classes', 'ner_tag', 'ner_tags', 'ner', 'entity'] |  ['pos_tags', 'pos_tag', 'pos', 'part_of_speech'] | ['chunk_tags', 'chunk_tag'] |



In the harness, we specify the data input in the following way:

<div class="heading" id="NER"> For NER  </div>

```python
#Import Harness from the nlptest library
from nlptest import Harness
harness = Harness(
            task='ner',
            model='en_core_web_sm',
            config= 'sample_config.yml',
            hub = "spacy",
            data= 'sample.conll/sample.csv' #Either of the two formats can be specified.
         
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