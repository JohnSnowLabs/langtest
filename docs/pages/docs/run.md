---
layout: docs
seotitle: Running Test Cases | NLP Test | John Snow Labs
title: Running Test Cases
permalink: /docs/pages/docs/run
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The `run()` method is called after the `generate()` method and is used to run all the specified tests. It returns a pass/fail flag for each test.

```python 
h.run()
```

Once the tests have been run using the `run()` method, the results can be accessed using the `.generated_results()` method. 

```python 
h.generated_results()
```

This method returns the generated results in the form of a Pandas dataframe, which provides a convenient and easy-to-use format for working with the test results. You can use this method to quickly identify the test cases that failed and to determine where fixes are needed.

A generated results dataframe looks like this:

{:.table2}
| category  | test_type |  original | test_case | expected_result |  actual_result | pass |
| - | - | - | - | - | - | - |
|robustness| lowercase | I live in Berlin | i live in berlin | Berlin: LOC | | False |
|robustness| uppercase | I live in Berlin | I LIVE IN BERLIN | Berlin: LOC | BERLIN: LOC | True |

</div></div>