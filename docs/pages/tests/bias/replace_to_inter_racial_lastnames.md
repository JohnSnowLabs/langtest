
<div class="h3-box" markdown="1">

## Replace To Interracial Last Names

This test checks if the NLP model can handle input text if the input text has interracial last names.

**alias_name:** `replace_to_inter_racial_lastnames`

<i class="fa fa-info-circle"></i>
<em>This data was curated using 2021 US census survey data. To apply this test appropriately in other contexts, please adapt the [data dictionaries](https://github.com/JohnSnowLabs/nlptest/blob/main/nlptest/transform/utils.py).</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_inter_racial_lastnames:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Mr. Hope will be here soon.|Mr. Khalid will be here soon.|

</div>