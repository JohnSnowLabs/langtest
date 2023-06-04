
<div class="h3-box" markdown="1">

## Insert Abbreviations

This test replaces familiar words or expressions in texts with their abbreviations. These abbreviations are commonly used on social media platforms and some are generic. It evaluates the NLP model's ability to handle text with such abbreviations.

**alias_name:** `add_abbreviation`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_abbreviation:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Amazing food! Great service!|Amzn food! Gr8 service!|
|Make sure you've gone online to download one of the vouchers - it's definitely not worth paying full price for!|Make sure u've gone onl 2 d/l one of da vouchers - it's dfntly not worth paying full price 4!|

</div>