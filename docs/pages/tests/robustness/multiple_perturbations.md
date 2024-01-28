
<div class="h3-box" markdown="1">

## Multiple Perturbations 

The `multiple_perturbations` test combines multiple tests into a single test by applying a sequence of perturbations to transform the given sentences. These perturbations are applied in a specific sequence.

Please note that this test is only supported for the `text-classification`, `question-answering`, and `summarization` tasks.

**alias_name:** `multiple_perturbations`

</div><div class="h3-box" markdown="1">

**Config YAML format** :
```
multiple_perturbations:
    min_pass_rate: 0.60
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
    perturbations1:
        lowercase
        add_ocr_typo
        titlecase
```
<i class="fa fa-info-circle"></i>
<em>The perturbation set `perturbations1` follows the transformation order: `lowercase` ➜ `add_ocr_typo` ➜ `titlecase`</em>

<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `multiple_perturbations` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|I live in London, United Kingdom since 2019 .|I <span style="color:red">L1Ve I^N</span> London, United Kingdom <span style="color:red">Slnce</span> 2019 .|
|I can't move to the USA because they have an average of 1000 tornadoes a year, and I'm terrified of them.|I <span style="color:red">Can'T Movc T^O T^Ie</span> Usa Hccause Thev Liave An Average Of 1000 Tornadoes A Ycar, A^Nd I'M Terrified Of Th^M.</span>|

</div>