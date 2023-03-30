---
layout: docs
header: true
seotitle: Bias Tests | NLP Test | John Snow Labs
title: Bias
key: tests
permalink: /docs/pages/tests/bias
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

{% assign parent_path = "pages/tests/bias" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "bias/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}
