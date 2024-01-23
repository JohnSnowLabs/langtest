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

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "text-classification/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

</div>