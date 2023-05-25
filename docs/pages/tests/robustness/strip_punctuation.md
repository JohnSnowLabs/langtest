
<div class="h3-box" markdown="1">

## Strip Punctuation

This test checks if the NLP model can handle sentences without punctuations at the end.

**alias_name:** `strip_punctuation`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
strip_punctuation:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog|
|Good morning!|Good morning|

</div>
