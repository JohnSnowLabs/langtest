---
layout: docs
header: true
seotitle: Data Input | NLP Test | John Snow Labs
title: Data Input
key: docs-examples
permalink: /docs/pages/docs/data_input
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Supported data input formats are task-dependent. For `ner` and `text-classification`, the user is meant to provide a **`CoNLL`** or **`CSV`** dataset. For `question-answering` the user is meant to choose from a list of benchmark datasets.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL and CSV|
|**text-classification**     |CSV
|**question-answering**     |Select list of benchmark datasets

</div><div class="h3-box" markdown="1">

### NER

There are 2 options for datasets to test NER models: **`CoNLL`** or **`CSV`** datasets. Here are some details of what these may look like:

#### CoNLL Format for NER

```bash
LEICESTERSHIRE NNP B-NP B-ORG
TAKE           NNP I-NP O
OVER           IN  B-PP O
AT             NNP B-NP O
TOP            NNP I-NP O
AFTER          NNP I-NP O
INNINGS        NNP I-NP O
VICTORY        NNP I-NP O
```

</div><div class="h3-box" markdown="1">

#### CSV Format for NER

{:.table2}
| Supported "text" column names | Supported "ner" column names | Supported "pos" column names | Supported "chunk" column names | 
| - | - | 
| ['text', 'sentences', 'sentence', 'sample'] |  ['label', 'labels ', 'class', 'classes', 'ner_tag', 'ner_tags', 'ner', 'entity'] |  ['pos_tags', 'pos_tag', 'pos', 'part_of_speech'] | ['chunk_tags', 'chunk_tag'] |

</div><div class="h3-box" markdown="1">

#### Passing a NER Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the nlptest library
from nlptest import Harness
harness = Harness(task='ner',
                  model='en_core_web_sm',
                  config='config.yml',
                  hub='spacy',
                  data='sample.conll') #Either of the two formats can be specified.
```

</div><div class="h3-box" markdown="1">

### Text Classification

There is 1 option for datasets to test Text Classification models: **`CSV`** datasets. Here are some details of what these may look like:

#### CSV Format for Text Classification

Here's a sample dataset:

{:.table2}
| text | label  |  
| - | - | 
|I thoroughly enjoyed Manna from Heaven. The hopes and dreams and perspectives of each of the characters is endearing and we, the audience, get to know each and every one of them, warts and all. And the ending was a great, wonderful and uplifting surprise! Thanks for the experience; I'll be looking forward to more.| 1 |
|Absolutely nothing is redeeming about this total piece of trash, and the only thing worse than seeing this film is seeing it in English class. This is literally one of the worst films I have ever seen. It totally ignores and contradicts any themes it may present, so the story is just really really dull. Thank god the 80's are over, and god save whatever man was actually born as "James Bond III".| 0 |

For `CSV` files, we support different variations of the column names. They are shown below :

{:.table2}
| Supported "text" column names | Supported "label" column names   |  
| - | - | 
| ['text', 'sentences', 'sentence', 'sample'] | ['label', 'labels ', 'class', 'classes'] |

</div><div class="h3-box" markdown="1">

#### Passing a Text Classification Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
#Import Harness from the nlptest library
from nlptest import Harness
harness = Harness(task='text-classification',
                  model='mrm8488/distilroberta-finetuned-tweets-hate-speech',
                  config='config.yml',
                  hub ='huggingface',
                  data='sample.csv')
```

</div><div class="h3-box" markdown="1">

### Question Answering

To test Question Answering models, the user is meant to select a benchmark dataset from the following list:

#### Benchmark Datasets

{:.table2}
| Dataset  | Source | Description |
| - | - | - |
|**BoolQ** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Training, development & test set from the BoolQ dataset, containing 15,942 labeled examples
|**BoolQ-test** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Test set from the BoolQ dataset, containing 3,245 labeled examples
|**BoolQ-test-tiny** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Truncated version of the test set from the BoolQ dataset, containing 50 labeled examples
|**NQ-open** | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) | Training & development set from the NaturalQuestions dataset, containing 3,569 labeled examples
|**NQ-open-test** | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) | Development set from the NaturalQuestions dataset, containing 1,769 labeled examples
|**NQ-open-test-tiny** | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) | Training, development & test set from the NaturalQuestions dataset, containing 50 labeled examples

</div><div class="h3-box" markdown="1">

#### Passing a Question Answering Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
#Import Harness from the nlptest library
from nlptest import Harness
harness = Harness(task='question-answering',
                  model='gpt-3.5-turbo',
                  config='config.yml',
                  hub ='openai',
                  data='BoolQ-test')
```

</div></div>