---
layout: docs
header: true
seotitle: Representation Tests | LangTest | John Snow Labs
title: Representation
key: tests
permalink: /docs/pages/tests/representation
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/tests/representation" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "representation/" %}
        {% include_relative {{ file_name }} %}        
    {% endif %}
{% endfor %}

</div>