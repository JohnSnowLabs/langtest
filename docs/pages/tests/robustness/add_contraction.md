
<div class="h3-box" markdown="1">

## Add Contraction

This test checks if the NLP model can handle input text if the data uses contractions instead of expanded forms.

**alias_name:** `add_contraction`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_contraction:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `add_contraction` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.


</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|He is not a great chess player.|He <span style="color:red">isn't</span> a great chess player.|
|I will wash the car this afternoon.|<span style="color:red">I'll</span> wash the car this afternoon.|

</div>
