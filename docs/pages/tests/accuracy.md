---
layout: docs
header: true
seotitle: Accuracy Tests | NLP Test | John Snow Labs
title: Accuracy
key: tests
permalink: /docs/pages/tests/accuracy
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

{% assign parent_path = "pages/tests/accuracy" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "accuracy/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}
