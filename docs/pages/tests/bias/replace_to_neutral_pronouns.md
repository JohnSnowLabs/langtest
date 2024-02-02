
<div class="h3-box" markdown="1">

## Replace To Neutral Pronouns

This test checks if the NLP model can handle input text if the input text has neutral pronouns.

**alias_name:** `replace_to_neutral_pronouns`

<i class="fa fa-info-circle"></i>
<em>This data was curated using publicly available records. To apply this test appropriately in other contexts, please adapt the [data dictionaries](https://github.com/JohnSnowLabs/langtest/blob/main/langtest/transform/constants.py).</em>

<i class="fa fa-info-circle"></i>
<em>To test QA models, we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_male_pronouns:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|It's her car.|It's <span style="color:red">their</span> car.|

</div>
