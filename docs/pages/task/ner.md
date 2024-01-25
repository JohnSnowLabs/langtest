---
layout: docs
header: true
seotitle: Ner | LangTest | John Snow Labs
title: Ner
key: task
permalink: /docs/pages/task/ner
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/task/ner" %}

Named Entity Recognition (NER) is a natural language processing (NLP) technique that involves identifying and classifying named entities within text. Named entities are specific pieces of information such as names of people, places, organizations, dates, times, quantities, monetary values, percentages, and more. NER plays a crucial role in understanding and extracting meaningful information from unstructured text data.

The primary goal of Named Entity Recognition is to locate and label these named entities in a given text. By categorizing them into predefined classes, NER algorithms make it possible to extract structured information from text, turning it into a more usable format for various applications like information retrieval, text mining, question answering, sentiment analysis, and more.


</div>

<div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Accuracy**](/docs/pages/tests/test#accuracy-tests)            | CoNLL, CSV and HuggingFace Datasets |
| [**Bias**](/docs/pages/tests/test#bias-tests)                |  CoNLL, CSV and HuggingFace Datasets                               |
| [**Fairness**](/docs/pages/tests/test#fairness-test)          | CoNLL, CSV and HuggingFace Datasets                             |
| [**Robustness**](/docs/pages/tests/test#robustness-tests)          | CoNLL, CSV and HuggingFace Datasets |
| [**Representation**](/docs/pages/tests/test#representation-tests)      | CoNLL, CSV and HuggingFace Datasets |


To get more information about the supported data, click [here](/docs/pages/docs/data).

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "ner/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>