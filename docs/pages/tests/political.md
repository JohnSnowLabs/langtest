---
layout: docs
header: true
seotitle: Political Tests | LangTest | John Snow Labs
title: Political
key: tests
permalink: /docs/pages/tests/political
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/tests/political" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "political/" %}
        {% include_relative {{ file_name }} %}        
    {% endif %}
{% endfor %}

</div>