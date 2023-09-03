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

The provided code initializes an instance of the Harness class. It accepts a data parameter, which can be specified as a `dictionary` with the following attributes.

```python
{
   "data_source": "",
   "subset": "",
   "feature_column": "",
   "target_column": "",
   "split": "",
   "source": "huggingface"
}
```
<br/>

{:.table2}
| Key  | Description |
| - | - |
|**data_source**(mandatory)     |Represents the name of the dataset being used.|
|**subset**(optional)     |Indicates the subset of the dataset being considered.
|**feature_column**(optional)       |Specifies the column that contains the input features.
|**target_column**(optional)     |Represents the column that contains the target labels or categories.
|**split**(optional)       |Denotes which split of the dataset should be used.|
|**source**(optional)|Set to ‘huggingface’ when loading Hugging Face dataset.|

Supported `data_source` formats are task-dependent. The following table provides an overview of the compatible data sources for each specific task.

{:.table2}
| Task  | Supported Data Inputs |  
| - | - | 
|**ner**     |CoNLL, CSV and HuggingFace Datasets|
|**text-classification**     |CSV and HuggingFace Datsets
|**question-answering**     |Select list of benchmark datasets or HuggingFace Datsets
|**summarization**     |Select list of benchmark datasets or HuggingFace Datsets
|**toxicity**     |Select list of benchmark datasets
|**clinical-tests**     |Select list of curated datasets
|**disinformation-test**    |Select list of curated datasets

</div><div class="h3-box" markdown="1">

### NER

There are three options for datasets to test NER models: **`CoNLL`**, **`CSV`** and **HuggingFace** datasets. Here are some details of what these may look like:

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
                  model={'model': 'en_core_web_sm', 'hub':'spacy'},
                  data={"data_source":'test.conll'},
                  config='config.yml') #Either of the two formats can be specified.
```

#### Passing a Hugging Face Dataset for NER to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="ner",
                  model={"model": "en_core_web_sm", "hub": "spacy"},
                  data={"data_source":'wikiann',
                  "subset":"en",
                  "feature_column":"tokens",
                  "target_column":'ner_tags',
                  "split":"test",
                  "source": "huggingface"
                  })
```


</div><div class="h3-box" markdown="1">

### Text Classification

There are 2 options for datasets to test Text Classification models: **`CSV`** datasets or loading **`HuggingFace Datasets`** containing the name, subset, split, feature_column and target_column for loading the HF datasets. Here are some details of what these may look like:

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
                  model={'model': 'mrm8488/distilroberta-finetuned-tweets-hate-speech', 'hub':'huggingface'},
                  data={"data_source":'sample.csv'},
                  config='config.yml')
```

</div><div class="h3-box" markdown="1">

#### Passing a Hugging Face Dataset for Text Classification to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="text-classification",
                  model={'model': 'mrm8488/distilroberta-finetuned-tweets-hate-speech', 'hub':'huggingface'},
                  data={"data_source":'glue',
                  "subset":"sst2",
                  "feature_column":"sentence",
                  "target_column":'label',
                  "split":"train",
                  "source": "huggingface"
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
|**LogiQA-test** | [LogiQA](https://aclanthology.org/2020.findings-emnlp.301/) | Testing set from the LogiQA dataset, containing 1000 question answers examples.
|**LogiQA-test-tiny** | [LogiQA](https://aclanthology.org/2020.findings-emnlp.301/) | Truncated version of the test set from the LogiQA dataset, containing 50 question and answers examples.
|**ASDiv-test** | [ASDiv](https://arxiv.org/abs/2106.15772) | Testing set from the ASDiv dataset, containing 1000 question answers examples.
|**ASDiv-test-tiny** | [ASDiv](https://arxiv.org/abs/2106.15772) | Truncated version of the test set from the ASDiv dataset, containing 50 question and answers examples.
|**Bigbench-Abstract-narrative-understanding-test** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Testing set from the Bigbench/Abstract Narrative Understanding dataset, containing 1000 question answers examples.
|**Bigbench-Abstract-narrative-understanding-test-tiny** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Truncated version of the test set from the Bigbench/Abstract Narrative Understanding dataset, containing 50 question and answers examples.
|**Bigbench-DisambiguationQA-test** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Testing set from the Bigbench/DisambiguationQA dataset, containing 207 question answers examples.
|**Bigbench-DisambiguationQA-test-tiny** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Truncated version of the test set from the Bigbench/DisambiguationQA dataset, containing 50 question and answers examples.
|**Bigbench-DisflQA-test** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Testing set from the Bigbench/DisflQA dataset, containing 1000 question answers examples.
|**Bigbench-DisflQA-test** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Truncated version of the test set from the Bigbench/DisflQA dataset, containing 50 question and answers examples.
|**Bigbench-Causal-judgment-test** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Testing set from the Bigbench/Causal Judgment dataset, containing 190 question and answers examples.
|**Bigbench-Causal-judgment-test-tiny** | [Bigbench Dataset](https://arxiv.org/abs/2206.04615) | Truncated version of the test set from the Bigbench/Causal Judgment dataset, containing 50 question and answers examples.


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
|**LogiQA** |Evaluate your model's accuracy on Machine Reading Comprehension with Logical Reasoning questions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LogiQA_dataset.ipynb)|
|**ASDiv** |Evaluate your model's ability to answer questions given a conversational context, focusing on dialogue-based question-answering. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/ASDiv_dataset.ipynb)|
|**BigBench Abstract narrative understanding** |Evaluate your model's performance in selecting the most relevant proverb for a given narrative. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)|
|**BigBench Causal Judgment** |Evaluate your model's performance in measuring the ability to reason about cause and effect. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)|
|**BigBench DisambiguationQA** |Evaluate your model's performance on determining the interpretation of sentences containing ambiguous pronoun references. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)|
|**BigBench DisflQA** |Evaluate your model's performance in picking the correct answer span from the context given the disfluent question. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)|

</div><div class="h3-box" markdown="1">

#### Passing a Question Answering Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='question-answering',
                  model={'model': 'text-davinci-003', 'hub':'openai'}, 
                  data={"data_source":'BoolQ-test'}, 
                  config='config.yml')
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
|**XSum** | Evaluate your model's ability to generate concise and informative summaries for long articles with the XSum dataset. It consists of articles and corresponding one-sentence summaries, offering a valuable benchmark for text summarization models. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/XSum_dataset.ipynb)|

</div><div class="h3-box" markdown="1">

#### Passing a Summarization Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='summarization',
                  model={'model': 'text-davinci-003', 'hub':'openai'}, 
                  data={"data_source":'XSum-test-tiny'}, 
                  config='config.yml')
```

#### Passing a Hugging Face Dataset for Summarization to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="summarization", 
                  model={'model': 'text-davinci-003', 'hub':'openai'}, 
                  data={"data_source":'samsum',
                  "feature_column":"dialogue",
                  "target_column":'summary',
                  "split":"test",
                  "source": "huggingface"
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
                  model={'model': 'text-davinci-003', 'hub':'openai'}, 
                  data={"data_source":'toxicity-test-tiny'})
```

</div><div class="h3-box" markdown="1">

### Disinformation Test

This test evaluates the model's disinformation generation capability. Users should choose a benchmark dataset from the provided list.

#### Datasets

{:.table2}
| Dataset  | Source | Description |
| - | - | - |
|**Narrative-Wedging** | [Truth, Lies, and Automation How Language Models Could Change Disinformation](https://cset.georgetown.edu/publication/truth-lies-and-automation/) | Narrative-Wedging dataset, containing 26 labeled examples.

</div><div class="h3-box" markdown="1">

#### Disinformation Test Dataset: Use Cases and Evaluations

{:.table2}
| Dataset  | Use Case |Notebook|
|-|
|**Narrative-Wedging** | Assess the model’s capability to generate disinformation targeting specific groups, often based on demographic characteristics such as race and religion. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Disinformation_Test.ipynb)

</div><div class="h3-box" markdown="1">

#### Passing a Disinformation Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='disinformation-test',
                  model={"model": "j2-jumbo-instruct", "hub":"ai21"},
                  data={"data_source": "Narrative-Wedging"})
```

</div></div>