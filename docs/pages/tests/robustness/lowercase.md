
<div class="h3-box" markdown="1">

## Lowercase

This test checks if the NLP model can handle input text that is in all lowercase. Like the uppercase test, this is important to ensure that your NLP model can handle input text in any case.

**alias_name:** `lowercase`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
lowercase:
    min_pass_rate: 0.8
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|the quick brown fox jumps over the lazy dog.|
|I AM VERY QUIET.|i am very quiet.|

</div>
