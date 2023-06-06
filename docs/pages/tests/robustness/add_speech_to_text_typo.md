
<div class="h3-box" markdown="1">

## Add Speech to Text Typo

This test evaluates the NLP model's proficiency in handling input text that contains common typos resulting from Speech to Text conversion. A Speech to Text typo dictionary is utilized to apply the most frequent typos found in speech recognition output to the input data.

**alias_name:** `add_speech_to_text_typo`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_speech_to_text_typo:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Andrew finally returned the French book to Chris that I bought last week.|Andrew finally returned the French book to Chris that I bot lass week.|
|The more you learn, the more you grow.|Thee morr you learn, the morr you grow.|

</div>