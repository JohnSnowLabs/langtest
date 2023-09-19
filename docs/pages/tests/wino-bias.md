---
layout: docs
header: true
seotitle: Wino Bias | LangTest | John Snow Labs
title:  Wino Bias 
key: tests
permalink: /docs/pages/tests/wino-bias
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

{% assign parent_path = "pages/tests/wino-bias" %}
{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "wino-bias/" %}
        {% include_relative {{ file_name }} %}        
    {% endif %}
{% endfor %}

</div>