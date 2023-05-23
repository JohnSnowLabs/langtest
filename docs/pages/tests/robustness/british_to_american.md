
<div class="h3-box" markdown="1">

## British to American

This test checks if the NLP model can handle input text with American accent. An accent dictionary is used to convert sentences into American accent.

**alias_name:** `british_to_american`

<i class="fa fa-info-circle"></i>

<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
british_to_american:
    min_pass_rate: 0.7
```

- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The technician analyzed your samples.|The technician analyzed your samples.|
|What colour is this?|What color is this?|

</div>
