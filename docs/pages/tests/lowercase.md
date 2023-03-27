---
layout: docs
header: true
seotitle: NLP Test | John Snow Labs
title: Lowercase
key: tests
permalink: /docs/pages/tests/lowercase
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text that is in all lowercase. Like the uppercase test, this is important to ensure that your NLP model can handle input text in any case.

**alias_name:** `lowercase`

</div><div class="h3-box" markdown="1">

## Config
```yaml
lowercase:
    min_pass_rate: <float>
```
**min_pass_rate:** Minimum pass rate to pass the test.

## Examples

{:.table2}
|Original|Testcase|
|-|
|The quick brown fox jumps over the lazy dog.|the quick brown fox jumps over the lazy dog.|
|I AM VERY QUIET.|i am very quiet.|


</div></div>