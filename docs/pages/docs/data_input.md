---
layout: docs
header: true
seotitle: Data Inputs | NLP Test | John Snow Labs
title: Data Inputs
key: docs-examples
permalink: /docs/pages/docs/data_input
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Supported data input formats are **`CoNLL`** and **`CSV`**. CoNLL dataset can only be loaded for `ner`. For `text-classification`, both formats are supported provided the column names are from a list of supported column names.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL and CSV|
|**text-classification**     |CSV

</div><div class="h3-box" markdown="1">

### Sample CoNLL format

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

</div><div class="h3-box" markdown="1">

### Sample CSV format

A sample CSV data input looks like the following : 

{:.table2}
| text | label  |  
| - | - | 
|I thoroughly enjoyed Manna from Heaven. The hopes and dreams and perspectives of each of the characters is endearing and we, the audience, get to know each and every one of them, warts and all. And the ending was a great, wonderful and uplifting surprise! Thanks for the experience; I'll be looking forward to more.| 1 |
|Absolutely nothing is redeeming about this total piece of trash, and the only thing worse than seeing this film is seeing it in English class. This is literally one of the worst films I have ever seen. It totally ignores and contradicts any themes it may present, so the story is just really really dull. Thank god the 80's are over, and god save whatever man was actually born as "James Bond III".| 0 |

For `CSV` files, we support different variations of the column names. They are shown below :

</div><div class="h3-box" markdown="1">

### For Text-Classification

{:.table2}
| Supported "text" column names | Supported "label" column names   |  
| - | - | 
| ['text', 'sentences', 'sentence', 'sample'] | ['label', 'labels ', 'class', 'classes'] |

</div><div class="h3-box" markdown="1">

### For NER

{:.table2}
| Supported "text" column names | Supported "ner" column names | Supported "pos" column names | Supported "chunk" column names | 
| - | - | 
| ['text', 'sentences', 'sentence', 'sample'] |  ['label', 'labels ', 'class', 'classes', 'ner_tag', 'ner_tags', 'ner', 'entity'] |  ['pos_tags', 'pos_tag', 'pos', 'part_of_speech'] | ['chunk_tags', 'chunk_tag'] |


In the harness, we specify the data input in the following way:

</div><div class="h3-box" markdown="1">

### For NER

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

</div><div class="h3-box" markdown="1">

### For Text-Classification

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