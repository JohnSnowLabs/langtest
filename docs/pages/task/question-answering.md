---
layout: docs
header: true
seotitle: Question Answering | LangTest | John Snow Labs
title: Question Answering
key: task
permalink: /docs/pages/task/question-answering
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

Question Answering (QA) models excel in extracting answers from a provided text based on user queries. These models prove invaluable for efficiently searching for answers within documents. Notably, some QA models exhibit the ability to generate answers independently, without relying on specific contextual information.

![Question Answering](/assets/images/task/question-answering.gif)

However, alongside their remarkable capabilities, there are inherent challenges in evaluating the performance of LLMs in the context of QA. To evaluate Language Models (LLMs) for the Question Answering (QA) task, LangTest provides the following test categories:

</div><div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Robustness**](/docs/pages/tests/test#robustness-tests)          | [Benchmark datasets](/docs/pages/benchmarks/benchmark), CSV, and HuggingFace Datasets |
| [**Bias**](/docs/pages/tests/test#bias-tests)                |  BoolQ (split: bias)                               |
| [**Accuracy**](/docs/pages/tests/test#accuracy-tests)            | [Benchmark datasets](/docs/pages/benchmarks/benchmark), CSV, and HuggingFace Datasets |
| [**Fairness**](/docs/pages/tests/test#fairness-tests)          | [Benchmark datasets](/docs/pages/benchmarks/benchmark), CSV, and HuggingFace Datasets     |
| [**Representation**](/docs/pages/tests/test#representation-tests)      | [Benchmark datasets](/docs/pages/benchmarks/benchmark), CSV, and HuggingFace Datasets |
| [**Factuality**](/docs/pages/tests/test#factuality-test)          | Factual-Summary-Pairs                             |
| [**Ideology**](/docs/pages/tests/test#ideology-tests)            | Curated list                                     |
| [**Legal**](/docs/pages/tests/test#legal-tests)               | Legal-Support                                    |
| [**Sensitivity**](/docs/pages/tests/test#sensitivity-tests)         | NQ-Open, OpenBookQA, wikiDataset                  |
| [**Stereoset**](/docs/pages/tests/test#stereoset-tests)           | StereoSet                                        |
| [**Sycophancy**](/docs/pages/tests/test#sycophancy-tests)          | synthetic-math-data, synthetic-nlp-data          |


To get more information about the supported data, click [here](/docs/pages/docs/data).

</div><div class="h3-box" markdown="1">

#### Task Specification

When specifying the task for Named Entity Recognition (NER), use the following format:

**task**: `Union[str, dict]`


For  accuracy, bias, fairness, and robustness we specify the task as a string

```python
task = "question-answering"
```

If you want to access some sub-task from `question-answering`, then you need to give the task as a dictionary.

```python
task = {"task" : "question-answering", "category":"sycophancy-test" }
```


{% assign parent_path = "pages/task/question-answering" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "question-answering/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>