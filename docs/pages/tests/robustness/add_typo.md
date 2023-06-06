
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
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The wuick brown fox jumps over the fazy dog.|
|Good morning|Good morninh|

</div>
