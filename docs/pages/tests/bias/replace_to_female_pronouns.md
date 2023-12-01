
<div class="h3-box" markdown="1">

## Replace To Female Pronouns

This test checks if the NLP model can handle input text if the input text has female pronouns.

**alias_name:** `replace_to_female_pronouns`

<i class="fa fa-info-circle"></i>
<em>This data was curated using publicly available records. To apply this test appropriately in other contexts, please adapt the [data dictionaries](https://github.com/JohnSnowLabs/langtest/blob/main/langtest/transform/utils.py).</em>

<i class="fa fa-info-circle"></i>
<em>To test QA models, we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_female_pronouns:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|He is brilliant.|She is brilliant.|
|He forgot his keys at the office.|She forgot her keys at the office.|

</div>
