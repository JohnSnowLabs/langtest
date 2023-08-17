---
layout: docs
header: true
seotitle: Clinical Tests | LangTest | John Snow Labs
title: Clinical Tests
key: tests
permalink: /docs/pages/tests/clinical-tests
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2023-08-17"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/tests/clinical-tests" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "clinical-tests/" %}
        {% include_relative {{ file_name }} %}        
    {% endif %}
{% endfor %}

</div>