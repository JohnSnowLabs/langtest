
<div class="h3-box" markdown="1">

## Number To Word

This test provides a convenient way to convert numerical values in text into their equivalent words. It is particularly useful for applications that require handling or processing textual data containing numbers.

**alias_name:** `number_to_word`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
number_to_word:
    min_pass_rate: 0.8
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `number_to_word` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|I live in London, United Kingdom since 2019 | I live in London, United Kingdom since <span style="color:red">two thousand and nineteen</span>|
|I can't move to the USA because they have an average of 100 tornadoes a year, and I'm terrified of them|I can't move to the USA because they have an average of <span style="color:red">one hundred</span> tornadoes a year, and I'm terrified of them|

</div>
