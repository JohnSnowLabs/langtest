---
layout: docs
header: true
seotitle: Fill Mask | LangTest | John Snow Labs
title: Fill Mask
key: task
permalink: /docs/pages/task/fill-mask
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

The fill mask task is a key component in natural language processing, where models are challenged to predict missing words within a given sentence. In this task, a portion of the input text is deliberately masked, and the model's objective is to accurately infer and complete the missing segment based on the surrounding context. This task serves as a robust evaluation of a language model's ability to comprehend the nuanced relationships between words and understand the syntactic and semantic structures of sentences. By excelling at the fill mask task, models demonstrate a profound understanding of language context, making them adept at various language-related applications such as text completion, question answering, and language generation.

</div>

<div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Stereotype**](/docs/pages/tests/test#Stereotype-tests)          | Wino-Bias, CrowS Pairs          |


To get more information about the supported data, click [here](/docs/pages/docs/data).

{% assign parent_path = "pages/task/fill-mask" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "fill-mask/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>