
<div class="h3-box" markdown="1">

## Replace To Sikh Names

This test checks if the NLP model can handle input text if the input text has Sikh names.

**alias_name:** `replace_to_sikh_names`

<i class="fa fa-info-circle"></i>
<em>This data was curated using [Kidpaw](https://www.kidpaw.com/). Please adapt the [data dictionaries](https://github.com/JohnSnowLabs/nlptest/blob/main/nlptest/transform/utils.py) to fit your use-case.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_sikh_names:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Billy will be here soon.|Armin will be here soon.|

</div>