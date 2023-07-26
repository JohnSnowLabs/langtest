---
layout: docs
header: true
seotitle: Data | LangTest | John Snow Labs
title: Data
key: docs-examples
permalink: /docs/pages/docs/data
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Supported data input formats are task-dependent. For `ner` and `text-classification`, the user is meant to provide a **`CoNLL`** or **`CSV`** dataset. For `question-answering`, `summarization` and `toxicity`  the user is meant to choose from a list of benchmark datasets we support.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL and CSV|
|**text-classification**     |CSV or a Dictionary (containing the name, subset, split, feature_column and target_column for loading the HF dataset.)
|**question-answering**     |Select list of benchmark datasets
|**summarization**     |Select list of benchmark datasets
|**toxicity**     |Select list of benchmark datasets

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
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='ner',
                  model='en_core_web_sm',
                  config='config.yml',
                  hub='spacy',
                  data='sample.conll') #Either of the two formats can be specified.
```

</div><div class="h3-box" markdown="1">

### Text Classification

There are 2 options for datasets to test Text Classification models: **`CSV`** datasets or a **`Dictionary`** containing the name, subset, split, feature_column and target_column for loading the HF datasets. Here are some details of what these may look like:

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

#### Passing a CSV Text Classification Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='text-classification',
                  model='mrm8488/distilroberta-finetuned-tweets-hate-speech',
                  config='config.yml',
                  hub ='huggingface',
                  data='sample.csv')
```

</div><div class="h3-box" markdown="1">

#### Dictionary Format for Text Classification
To handle text classification task for Hugging Face Datasets, the Harness class accepts the data parameter as a dictionary with following attributes:

<i class="fa fa-info-circle"></i>
<em>It's important to note that the default values for the **`split`**, **`feature_column`**, and **`target_column`** attributes are **`test`**, **`text`**, and **`label`**, respectively.</em>

```python
{
   "name": "",
   "subset": "",
   "feature_column": "",
   "target_column": "",
   "split": ""
}
```

#### Passing a Hugging Face Dataset for Text Classification to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="text-classification", hub="huggingface",
                  model="distilbert-base-uncased-finetuned-sst-2-english",
                  data={"name":'glue',
                  "subset":"sst2",
                  "feature_column":"sentence",
                  "target_column":'label',
                  "split":"train"
                  })
```

</div><div class="h3-box" markdown="1">

### Question Answering

To test Question Answering models, the user is meant to select a benchmark dataset from the following list:

#### Benchmark Datasets

{:.table2}
| Dataset  | Source | Description |
| - | - | - |
|**BoolQ** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Training, development & test set from the BoolQ dataset, containing 15,942 labeled examples
|**BoolQ-dev** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Dev set from the BoolQ dataset, containing 3,270 labeled examples
|**BoolQ-dev-tiny** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Truncated version of the dev set from the BoolQ dataset, containing 50 labeled examples
|**BoolQ-test** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Test set from the BoolQ dataset, containing 3,245 labeled examples. This dataset does not contain labels and accuracy & fairness tests cannot be run with it.
|**BoolQ-test-tiny** | [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) | Truncated version of the test set from the BoolQ dataset, containing 50 labeled examples. This dataset does not contain labels and accuracy & fairness tests cannot be run with it.
|**NQ-open** | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) | Training & development set from the NaturalQuestions dataset, containing 3,569 labeled examples
|**NQ-open-test** | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) | Development set from the NaturalQuestions dataset, containing 1,769 labeled examples
|**NQ-open-test-tiny** | [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) | Training, development & test set from the NaturalQuestions dataset, containing 50 labeled examples
|**TruthfulQA** | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/) | Training, test set from the TruthfulQA dataset, containing 817 questions that span 38 categories, including health, law, finance and politics.
|**TruthfulQA-test** | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/) | Testing set from the TruthfulQA dataset, containing 164 question and answer examples.
|**TruthfulQA-test-tiny** | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/) | Truncated version of TruthfulQA dataset which contains 50 question answer examples
|**MMLU-test** | [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) | Test set from the MMLU dataset which covers 57 tasks including elementary mathematics, US history, computer science, law, and more. We took 50 samples from each tasks in the test set.
|**MMLU-test-tiny** | [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) | Truncated version of test set from the MMLU dataset which covers 57 tasks including elementary mathematics, US history, computer science, law, and more. We took 10 samples from each tasks in the test-tiny set.
|**NarrativeQA-test** | [The NarrativeQA Reading Comprehension Challenge](https://aclanthology.org/Q18-1023/) | Testing set from the NarrativeQA dataset, containing 3000 stories and corresponding questions designed to test reading comprehension, especially on long documents.
|**NarrativeQA-test-tiny** | [The NarrativeQA Reading Comprehension Challenge](https://aclanthology.org/Q18-1023/) | Truncated version of NarrativeQA dataset which contains 50 stories and corresponding questions examples.
|**HellaSwag-test** | [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472/) | Dev set Training set from the hellaswag dataset with 3000 examples which is a benchmark for Commonsense NLI. It includes a context and some endings which complete the context.
|**HellaSwag-test-tiny** | [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472/) | Truncated version of the test set from the hellaswag dataset with 50 examples.
|**Quac-test** | [Quac: Question Answering in Context](https://aclanthology.org/D18-1241/) | Testing set from the QuAC dataset with 1000 examples for modeling, understanding, and participating in information seeking dialog.
|**Quac-test-tiny** | [Quac: Question Answering in Context](https://aclanthology.org/D18-1241/) | Truncated version of the val set from the QuAC dataset with 50 examples.
|**OpenBookQA-test** | [OpenBookQA Dataset](https://allenai.org/data/open-book-qa) | Testing set from the OpenBookQA dataset, containing 500 multiple-choice elementary-level science questions
|**OpenBookQA-test-tiny** | [OpenBookQA Dataset](https://allenai.org/data/open-book-qa) | Truncated version of the test set from the OpenBookQA dataset, containing 50 multiple-choice examples.
|**BBQ-test** | [BBQ Dataset: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/abs/2110.08193) | Testing set from the BBQ dataset, containing 1000 question answers examples.
|**BBQ-test-tiny** | [BBQ Dataset: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/abs/2110.08193) | Truncated version of the test set from the BBQ dataset, containing 50 question and answers examples.

</div><div class="h3-box" markdown="1">

#### Comparing Question Answering Benchmarks: Use Cases and Evaluations

Langtest comes with different datasets to test your models, covering a wide range of use cases and evaluation scenarios.


{:.table2}
| Dataset  | Use Case |Notebook|
|-|
|**BoolQ** | Evaluate the ability of your model to answer boolean questions (yes/no) based on a given passage or context. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/BoolQ_dataset.ipynb)|
|**NQ-open** |Evaluate the ability of your model to answer open-ended questions based on a given passage or context. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/NQ_open_dataset.ipynb)|
|**TruthfulQA** |Evaluate the model's capability to answer questions accurately and truthfully based on the provided information.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/TruthfulQA_dataset.ipynb)|
|**MMLU** |Evaluate language understanding models' performance in the different domain. It covers 57 subjects across STEM, the humanities, the social sciences, and more.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/mmlu_dataset.ipynb)|
|**NarrativeQA** |Evaluate your model's ability to comprehend and answer questions about long and complex narratives, such as stories or articles.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/NarrativeQA_Question_Answering.ipynb)|
|**HellaSwag** |Evaluate your model's ability in completions of sentences.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/HellaSwag_Question_Answering.ipynb)|
|**Quac** |Evaluate your model's ability to answer questions given a conversational context, focusing on dialogue-based question-answering. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/quac_dataset.ipynb)|
|**OpenBookQA** |Evaluate your model's ability to answer questions that require complex reasoning and inference based on general knowledge, similar to an "open-book" exam.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/quac_dataset.ipynb)|
|**BBQ** |Evaluate how your model respond to questions in the presence of social biases against protected classes across various social dimensions. Assess biases in model outputs with both under-informative and adequately informative contexts, aiming to promote fair and unbiased question answering models| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/BBQ_dataset.ipynb)|

</div><div class="h3-box" markdown="1">

#### Passing a Question Answering Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='question-answering',
                  model='gpt-3.5-turbo',
                  config='config.yml',
                  hub ='openai',
                  data='BoolQ-test')
```

</div><div class="h3-box" markdown="1">

### Summarization

To test Summarization models, the user is meant to select a benchmark dataset from the following list:

#### Benchmark Datasets

{:.table2}
| Dataset  | Source | Description |
| - | - | - |
|**XSum** | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclanthology.org/D18-1206/) | Training & development set from the Extreme Summarization (XSum) Dataset, containing 226,711 labeled examples
|**XSum-test** | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclanthology.org/D18-1206/) | Test set from the Xsum dataset, containing 1,000 labeled examples
|**XSum-test-tiny** | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclanthology.org/D18-1206/) | Truncated version of the test set from the Xsum dataset, containing 50 labeled examples

</div><div class="h3-box" markdown="1">
#### Summarization Benchmarks: Use Cases and Evaluations

{:.table2}
| Dataset  | Use Case |Notebook|
|-|
|**XSum** | Evaluate your model's ability to generate concise and informative summaries for long articles with the XSum dataset. It consists of articles and corresponding one-sentence summaries, offering a valuable benchmark for text summarization models. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/OpenAI_QA_Testing_Notebook.ipynb)|

</div><div class="h3-box" markdown="1">

#### Passing a Summarization Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='summarization',
                  model='text-davinci-002',
                  config='config.yml',
                  hub ='openai',
                  data='XSum-test-tiny')
```

#### Passing a Hugging Face Dataset for Summarization to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="summarization", 
                  hub="openai",
                  model="text-davinci-003",
                  data={"name":'samsum',
                  "feature_column":"dialogue",
                  "target_column":'summary',
                  "split":"test"
                  })
```
</div><div class="h3-box" markdown="1">

### Toxicity

This test checks the toxicity of the completion., the user is meant to select a benchmark dataset from the following list:

#### Benchmark Datasets

{:.table2}
| Dataset  | Source | Description |
| - | - | - |
|**toxicity-test-tiny** | [Real Toxicity Prompts](https://aclanthology.org/2020.findings-emnlp.301/) | Truncated set from the Real Toxicity Prompts Dataset, containing 80 examples.

</div><div class="h3-box" markdown="1">

#### Toxicity Benchmarks: Use Cases and Evaluations

{:.table2}
| Dataset  | Use Case |Notebook|
|-|
|**Real Toxicity Prompts** | Evaluate your model's accuracy in recognizing and handling toxic language with the Real Toxicity Prompts dataset. It contains real-world prompts from online platforms, ensuring robustness in NLP models to maintain safe environments. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/OpenAI_QA_Testing_Notebook.ipynb)

</div><div class="h3-box" markdown="1">

#### Passing a Toxicity Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='toxicity',
                  model='text-davinci-002',
                  hub='openai',
                  data='toxicity-test-tiny')
```

</div></div>