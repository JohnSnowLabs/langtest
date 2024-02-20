
<div class="h3-box" markdown="1">

## ADD Abbreviations

This test replaces familiar words or expressions in texts with their abbreviations. These abbreviations are commonly used on social media platforms and some are generic. It evaluates the NLP model's ability to handle text with such abbreviations.

**alias_name:** `add_abbreviation`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

#### Config
```yaml
add_abbreviation:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `add_abbreviation` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Amazing food! Great service!|<span style="color:red">Amzn food! Gr8</span> service!|
|Make sure you've gone online to download one of the vouchers - it's definitely not worth paying full price for!|Make sure <span style="color:red">u've</span> gone <span style="color:red">onl 2 d/l</span> one of <span style="color:red">da</span> vouchers - it's <span style="color:red">dfntly</span> not worth paying full price <span style="color:red">4</span>!|

</div>