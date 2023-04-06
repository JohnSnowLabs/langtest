
## Replace To High Income Country

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text if the input text has countries with high income.

**alias_name:** `replace_to_high_income_country`
<em>This data was curated from World Bank. To apply this test appropriately in other contexts, please adapt the [data dictionaries.](https://github.com/JohnSnowLabs/nlptest/blob/main/nlptest/transform/utils.py)</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_high_income_country:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Indonesia is one of the most populated countries.|U.S. is one of the most populated countries.|


</div></div>