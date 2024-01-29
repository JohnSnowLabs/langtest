---
layout: docs
header: true
seotitle: Summarization | LangTest | John Snow Labs
title: Summarization
key: task
permalink: /docs/pages/task/summarization
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">


A summarization task involves condensing a larger piece of text into a shorter version while preserving its key information and main points. This process requires understanding the content, identifying essential details, and expressing them concisely and coherently. Summarization tasks can range from simple sentence compression to creating abstracts of research papers or distilling the main arguments of a complex document. They play a crucial role in information retrieval, helping readers quickly grasp the essence of a text without having to go through the entire document.

</div><div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Accuracy**](/docs/pages/tests/test#accuracy-tests)            | [XSum](/docs/pages/benchmarks/other_benchmarks/xsum), [MultiLexSum](/docs/pages/benchmarks/legal/multilexsum) |
| [**Bias**](/docs/pages/tests/test#bias-tests)                |  [XSum](/docs/pages/benchmarks/other_benchmarks/xsum), [MultiLexSum](/docs/pages/benchmarks/legal/multilexsum)                               |
| [**Fairness**](/docs/pages/tests/test#fairness-test)          | [XSum](/docs/pages/benchmarks/other_benchmarks/xsum), [MultiLexSum](/docs/pages/benchmarks/legal/multilexsum)                             |
| [**Robustness**](/docs/pages/tests/test#robustness-tests)          | [XSum](/docs/pages/benchmarks/other_benchmarks/xsum), [MultiLexSum](/docs/pages/benchmarks/legal/multilexsum) |
| [**Representation**](/docs/pages/tests/test#representation-tests)      | [XSum](/docs/pages/benchmarks/other_benchmarks/xsum), [MultiLexSum](/docs/pages/benchmarks/legal/multilexsum) |

{% assign parent_path = "pages/task/summarization" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "summarization/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>