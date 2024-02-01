
<div class="h3-box" markdown="1">


## Add Negation

The Evaluating Model's Sensitivity to Negation Test focuses on assessing a model's responsiveness to negations introduced into its input text. The primary objective is to determine whether the model can effectively detect and respond to negations. This evaluation consists of two key stages: first, introducing negations into the input text, particularly after verbs like "is," "was," "are," and "were"; and second, observing how the model behaves or how sensitive it is when presented with input containing these negations.

**alias_name:** `add_negation`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
  sensitivity:
    add_negation:
      min_pass_rate: 0.70
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Example

{:.table3}
| original                                           | test_case                                              | expected_result                                                          | actual_result                                                       |   eval_score |   pass |
|:---------------------------------------------------|:-------------------------------------------------------|:-------------------------------------------------------------------------|:--------------------------------------------------------------------|-------------:|-------:|
| what is the name of the hat you wear at graduation | what is <span style="color:red">not</span> the name of the hat you wear at graduation | The hat typically worn at graduation ceremonies is called a mortarboard. | A mortarboard is the name of the hat worn at graduation ceremonies. |    0.0287267 |      False |




</div>