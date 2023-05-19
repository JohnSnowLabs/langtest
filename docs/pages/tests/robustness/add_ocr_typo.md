
<div class="h3-box" markdown="1">

## Add Ocr Typo

This test checks if the NLP model can handle input text with common ocr typos. A ocr typo dictionary is used to apply most common ocr typos to the input data.

**alias_name:** `add_ocr_typo`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_ocr_typo:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|This organization's art can win tough acts.|Tbis organization's a^rt c^an w^in tougb acts.|
|Anyone can join our community garden.|Anyone c^an j0in o^ur communitv gardcn.|

</div>