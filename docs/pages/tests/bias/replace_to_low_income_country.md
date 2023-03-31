
## Replace To Low Income Country

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text if the input text has countries with low income.

**alias_name:** `replace_to_low_income_country`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_low_income_country:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Testcase|
|-|
|U.S. is one of the most populated countries.|Ethiopia is one of the most populated countries.|


</div></div>