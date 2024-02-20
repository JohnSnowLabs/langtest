
<div class="h3-box" markdown="1">

## Add Typo

This test checks if the NLP model can handle input text with typos. A typo frequency dictionary is used to apply most common typos to the input data.

**alias_name:** `add_typo`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_typo:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
    parameters:
      count: 1 # Defaults to 1
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `add_typo` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.
- **count (int):** Number of variations of sentence to be constructed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The <span style="color:red">wuick</span> brown fox jumps over the <span style="color:red">fazy</span> dog.|
|Good morning|Good <span style="color:red">morninh</span>|

</div>
