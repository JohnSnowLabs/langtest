---
layout: docs
header: true
seotitle: Text Generation | LangTest | John Snow Labs
title: Text Generation
key: task
permalink: /docs/pages/task/text-generation
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

Text generation is a natural language processing (NLP) task focused on creating coherent and contextually relevant sequences of words or sentences. In this task, models are trained to generate human-like text based on a given prompt, context, or input. Unlike tasks such as classification or translation, text generation involves creating new content rather than selecting from predefined options. This task often utilizes generative language models, such as GPT (Generative Pre-trained Transformer), to produce diverse and contextually appropriate responses.

</div>

<div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Clinical**](/docs/pages/tests/test#clinical-test)                |  Medical-files, Gastroenterology-files, Oromaxillofacial-files                           |
| [**Disinformation**](/docs/pages/tests/test#disinformation-test)      | Narrative-Wedging |
| [**Security**](/docs/pages/tests/test#security-test)            | Prompt-Injection-Attack |
| [**Toxicity**](/docs/pages/tests/test#toxicity-tests)          | Toxicity |

To get more information about the supported data, click [here](/docs/pages/docs/data#text-generation).

{% assign parent_path = "pages/task/text-generation" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "text-generation/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>