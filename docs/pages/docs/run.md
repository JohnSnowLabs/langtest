---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Running Test Cases
permalink: /docs/pages/docs/run
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Called after **.generate()** method and is to used to run all the specified tests. Returns a pass/fail flag for each test.

```python 
harness.run()
```

Once the tests have been run using the harness.run() method, the results can be accessed using the **`.generated_results()`** method. 
```python 
harness.generated_results()
```
This method returns the generated results in the form of a pandas dataframe, which provides a convenient and easy-to-use format for working with the test results. You can use this method to quickly identify the test cases that failed and to determine where fixes are needed.

 A sample generated results dataframe looks like the one given below:

{:.table2}
| category  | test_type |  original | test_case | expected_result |  actual_result | pass |
| - | - | - | - | - | - | - |
|robustness| lowercase | I live in Berlin | i live in berlin | [O, O, O, LOC] | [O, O, O, O] | False |
|robustness| uppercase | I live in Berlin | I LIVE IN BERLIN | [O, O, O, LOC] | [O, O, O, LOC] | True |

</div></div>