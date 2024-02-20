
<div class="h3-box" markdown="1">

## Swap Entities

This test shuffles the labeled entities in the input to test the models robustness.

**alias_name:** `swap_entities`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
swap_entities:
    min_pass_rate: 0.8
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
    parameters:
      count: 1 # Defaults to 1
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `swap_entities` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.
- **count (int):** Number of variations of sentence to be constructed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|I love Paris.|I love <span style="color:red">Istanbul.</span>|
|Jack is sick today.|<span style="color:red">Adam</span> is sick today.|

</div>
