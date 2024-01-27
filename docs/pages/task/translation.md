---
layout: docs
header: true
seotitle: Translation | LangTest | John Snow Labs
title: Translation
key: task
permalink: /docs/pages/task/translation
aside:
    toc: true
sidebar:
    nav: task
show_edit_on_github: true
nav_key: task
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/task/translation" %}


Translation is the process of converting a sequence of text from one language into another. It operates within the framework of a sequence-to-sequence problem, which is a versatile approach for generating output based on an input, such as translation or summarization. While translation systems primarily facilitate the conversion of written texts between different languages, they can also extend to transforming spoken language, encompassing tasks like text-to-speech or speech-to-text conversion. Translation plays a vital role in bridging linguistic barriers and facilitating communication across diverse cultural and linguistic contexts.


</div><div class="h3-box" markdown="1">

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Robustness**](/docs/pages/tests/test#robustness-tests)          | Translation |


To get more information about the supported data, click [here](/docs/pages/docs/data).

</div><div class="h3-box" markdown="1">

#### Task Specification

When specifying the task for Translation, use the following format:


**task**: `str`

```python
task = "translation"
```

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "translation/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>