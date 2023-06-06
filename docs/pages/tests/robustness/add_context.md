
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
    parameters:
      ending_context: ['Bye', 'Reported']
      starting_context: ['Hi', 'Good morning', 'Hello']
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **starting_context (<List[str]>):** Phrases to be added at the start of inputs.
- **ending_context (<List[str]>):** Phrases to be added at the end of inputs.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog, bye.|
|I love playing football.|Hello, I love playing football.|


</div>
