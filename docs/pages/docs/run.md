---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Running the testcases
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

To get the run results in the form of a pandas dataframe we can use the **.generated_results()** method.

```python 
harness.generated_results()
```

 It returns the generated results in the form of a dataframe with pass/fail flag for each test that we had specified.

 The generated results dataframe looks something like the following:

{:.table2}
| category  | test_type |  original | test_case | expected_result |  actual_result | pass |
| - | - | - | - | - | - | - |
|robustness| lowercase | I live in Berlin | i live in berlin | [O, O, O, LOC] | [O, O, O, O] | False |
|robustness| uppercase | I live in Berlin | I LIVE IN BERLIN | [O, O, O, LOC] | [O, O, O, LOC] | True |

</div></div>