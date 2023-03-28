
## Replace To Male Pronouns

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text if the input text has female pronouns.

**alias_name:** `replace_to_female_pronouns`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
replace_to_female_pronouns:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Testcase|
|-|
|He is brilliant.|She is brilliant.|
|It's his car.|It's her car.|


</div></div>