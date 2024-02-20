
<div class="h3-box" markdown="1">

## Adjective Synonym Swap

This test provides a convenient way to convert adjectives into their equivalent synonyms.

**alias_name:** `adjective_synonym_swap`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
adjective_synonym_swap:
    min_pass_rate: 0.8
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `adjective_synonym_swap` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Lisa is wearing a beautiful shirt today. This soup is not edible. | Lisa is wearing a <span style="color:red">pretty</span> shirt today. This soup is <span style="color:red">consumable.</span>|
|They have a beautiful house.|They have a <span style="color:red">alluring</span> house.|

</div>
