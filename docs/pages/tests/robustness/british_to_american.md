
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
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `british_to_american` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The technician analysed your samples.|The technician <span style="color:red">analyzed</span> your samples.|
|What colour is this?|What <span style="color:red">color</span> is this?|

</div>
