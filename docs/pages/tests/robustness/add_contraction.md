
## Add Contraction

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text if the data uses contractions instead of expanded forms.

**alias_name:** `add_contraction`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_contraction:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Testcase|
|-|
|He is not a great chess player.|He isn't a great chess player.|
|I will wash the car this afternoon.|I'll wash the car this afternoon.|


</div></div>