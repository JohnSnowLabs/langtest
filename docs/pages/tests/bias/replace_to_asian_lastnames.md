
## Replace To Asian Last Names

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text if the input text has asian last names.

**alias_name:** `replace_to_asian_lastnames`

<em>This data was curated using the 2021 US census data survey. To apply this test appropriately in other contexts, please adapt the [data dictionaries.](https://github.com/JohnSnowLabs/nlptest/blob/main/nlptest/transform/utils.py)</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_asian_lastnames:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Mr. Hope will be here soon.|Mr. Nhan will be here soon.|



</div></div>