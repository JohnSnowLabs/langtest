---
layout: docs
header: true
seotitle: Toxicity Tests | LangTest | John Snow Labs
title: Toxicity
key: tests
permalink: /docs/pages/tests/toxicity
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/tests/toxicity" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "toxicity/" %}
        {% include_relative {{ file_name }} %}        
    {% endif %}
{% endfor %}

</div>