
<div class="h3-box" markdown="1">

## Add Punctuation

This test checks if the NLP model can handle input text with sentences with a punctuation at the end. The added punctuation is randomly chosen from the list `['!', '?', ',', '.', '-', ':', ';']`. If there already is a punctuation it is not changed.

**alias_name:** `add_punctuation`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_punctuation:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `add_punctuation` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog|The quick brown fox jumps over the lazy dog<span style="color:red">.</span>|
|Good morning|Good morning<span style="color:red">!</span>|

</div>
