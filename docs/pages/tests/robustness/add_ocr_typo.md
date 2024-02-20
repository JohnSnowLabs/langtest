
<div class="h3-box" markdown="1">

## Add OCR Typo

This test checks if the NLP model can handle input text with common ocr typos. A ocr typo dictionary is used to apply most common ocr typos to the input data.

**alias_name:** `add_ocr_typo`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_ocr_typo:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
    parameters:
        count: 1 # Defaults to 1
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `add_ocr_typo` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.
- **count (int):** Number of variations of sentence to be constructed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|This organization's art can win tough acts.|<span style="color:red">Tbis</span> organization's <span style="color:red">a^rt c^an w^in tougb</span> acts.|
|Anyone can join our community garden.|Anyone <span style="color:red">c^an j0in o^ur communitv gardcn</span>.|

</div>