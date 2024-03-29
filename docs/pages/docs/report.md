---
layout: docs
seotitle: Retrieving Reports | LangTest | John Snow Labs
title: Retrieving Reports
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
A sample report looks like this:

{:.table2}
| category  | test_type |  fail_count | pass_count | pass_rate |  minimum_pass_rate | pass |
| - | - | - | - | - | - | - |
|robustness | lowercase | 77 | 14 | 15% | 60%  | False |
|robustness | uppercase | 11 | 80 | 88% | 60%  | True |


### Different Report formats


```python
# Generate Report as Dictionary and Save as Json Report
h.report(format="dict",save_dir="report.json")

# Generate and Save Excel Report
h.report(format="excel", save_dir="report.xlsx")

# Generate and Save HTML Report
h.report(format="html", save_dir="report.html")

# Generate and Save Text Report
h.report(format="text", save_dir="report.txt")

# Generate and Save Markdown Report
h.report(format="markdown", save_dir="report.md")
```
</div><div class="h3-box" markdown="1">

</div></div>