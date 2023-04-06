
## Replace To Jain Names

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text if the input text has Jain names.

**alias_name:** `replace_to_jain_names`

<i class="fa fa-info-circle"></i>
<em>This data was curated using [Kidpaw](https://www.kidpaw.com/). Please adapt the [data dictionaries](https://github.com/JohnSnowLabs/nlptest/blob/main/nlptest/transform/utils.py) to fit your use-case.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_jain_names:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Billy will be here soon.|Maulik will be here soon.|



</div></div>