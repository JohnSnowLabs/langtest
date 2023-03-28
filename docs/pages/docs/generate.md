---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Generating the testcases
permalink: /docs/pages/docs/generate
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The **generate()** method automatically generates the test cases (based on the provided configuration). The configuration for the tests can be passed in the form of a YAML file or using .configure() method.

Config YAML format :

```shell 

defaults:
  min_pass_rate: 0.65
tests:     
  robustness:
    lowercase:
      min_pass_rate: 0.60
    uppercase:
      min_pass_rate: 0.60
  
```

If config file not present, we can use the **.configure()** method to configure the harness to perform the needed tests.

```python
harness.configure(
{'defaults': {'min_pass_rate': 0.65},
 'tests': {'robustness': {'lowercase': {'min_pass_rate': 0.60}, 
                          'uppercase':{'min_pass_rate': 0.60}}
          }
 }
 )
```

**Generating test cases**:
```python
harness.generate()
```

After generating the testcases we can use the **.testcases()** method. 
```python
harness.testcases()
```
This method returns the produced testcases in form of a pandas data frame – making them easy to edit, filter, import, or export. We can manually review the list of generated test cases, and decide on which ones to keep or edit. 

The testcases dataframe looks something like the following:

{:.table2}
| category  | test_type |  original | test_case | expected_result | 
| - | - | - | - | - |
|robustness| lowercase | I live in Berlin | i live in berlin | [O, O, O, LOC] |
|robustness| uppercase | I live in Berlin | I LIVE IN BERLIN | [O, O, O, LOC] |


</div></div>