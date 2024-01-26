---
layout: docs
header: true
seotitle: Text Classification | LangTest | John Snow Labs
title: Text Classification
key: task
permalink: /docs/pages/task/text-classification
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/task/text-classification" %}


Text classification is a common task in natural language processing (NLP) where text is categorized into different labels or classes. Many major companies utilize text classification for various practical purposes, integrating it into their production systems. Among the most popular applications is sentiment analysis, which assigns labels like 🙂 positive, 🙁 negative, or 😐 neutral to text sequences.

</div>

<div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Accuracy**](/docs/pages/tests/test#accuracy-tests)            | CSV and HuggingFace Datasets |
| [**Bias**](/docs/pages/tests/test#bias-tests)                |  CSV and HuggingFace Datasets                               |
| [**Fairness**](/docs/pages/tests/test#fairness-test)          | CSV and HuggingFace Datasets                             |
| [**Robustness**](/docs/pages/tests/test#robustness-tests)          | CSV and HuggingFace Datasets |
| [**Representation**](/docs/pages/tests/test#representation-tests)      | CSV and HuggingFace Datasets |


To get more information about the supported data, click [here](/docs/pages/docs/data).


{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "text-classification/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>