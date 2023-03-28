---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: report()
permalink: /docs/pages/docs/report
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Called after harness.run() and it summarizes the results giving information about pass and fail counts and overall test pass/fail flag.

```python
harness.report()
```
The report looks something like the following:

{:.table2}
| category  | test_type |  fail_count | pass_count | pass_rate |  minimun_pass_rate | pass |
| - | - | - | - | - | - | - |
|Robustness | lowercase | 77 | 14 | 15% | 60%  | False |
|Robustness | uppercase | 11 | 80 | 88% | 60%  | True |



</div></div>