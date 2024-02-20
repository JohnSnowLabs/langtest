
<div class="h3-box" markdown="1">

## Add Context

This test checks if the NLP model can handle input text with added context, such as a greeting or closing.

**alias_name:** `add_context`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_context:
    min_pass_rate: 0.65
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
    parameters:
      ending_context: ['Bye', 'Reported']
      starting_context: ['Hi', 'Good morning', 'Hello']
      count: 1 # Defaults to 1
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `add_context` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **starting_context (<List[str]>):** Phrases to be added at the start of inputs.
- **ending_context (<List[str]>):** Phrases to be added at the end of inputs.
- **prob (float):** Controls the proportion of words to be changed.
- **count (int):** Number of variations of sentence to be constructed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog, <span style="color:red">bye.</span>|
|I love playing football.|<span style="color:red">Hello,</span> I love playing football.|


</div>
