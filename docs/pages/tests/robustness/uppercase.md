
<div class="h3-box" markdown="1">

## Uppercase

This test checks if the NLP model can handle input text that is in all uppercase. Accidentally entering text in all caps is common, and you want to ensure that your NLP model can still process it correctly.

**alias_name:** `uppercase`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
uppercase:
    min_pass_rate: 0.8
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `uppercase` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.|
|I like to shout.|I LIKE TO SHOUT.|

</div>
