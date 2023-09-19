---
layout: docs
header: true
seotitle: Sensitivity Test | LangTest | John Snow Labs
title: Sensitivity
key: tests
permalink: /docs/pages/tests/sensitivity
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/tests/sensitivity" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "sensitivity/" %}
        {% include_relative {{ file_name }} %}        
    {% endif %}
{% endfor %}

</div>