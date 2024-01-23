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

{% assign parent_path = "pages/task/question-answering" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "question-answering/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>