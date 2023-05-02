---
layout: docs
seotitle: Report | NLP Test | John Snow Labs
title: Report
permalink: /docs/pages/docs/report
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Called after the `run()` method, the `report()` method summarizes the results of your tests, providing information about the number of tests that passed and failed for each test category, as well as an overall pass/fail flag. 

It provides a convenient way to quickly evaluate the results of your tests and determine whether your model is performing as expected. By using this, you can identify areas where your model needs improvement and make necessary changes to ensure that it meets your requirements.

```python
h.report()
```
A sample report looks like the one given below:

{:.table2}
| category  | test_type |  fail_count | pass_count | pass_rate |  minimum_pass_rate | pass |
| - | - | - | - | - | - | - |
|robustness | lowercase | 77 | 14 | 15% | 60%  | False |
|robustness | uppercase | 11 | 80 | 88% | 60%  | True |



</div></div>