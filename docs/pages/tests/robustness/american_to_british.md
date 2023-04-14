
<div class="h3-box" markdown="1">

## American to British

This test checks if the NLP model can handle input text with british accent. An accent dictionary is used to convert sentences into british accent.

**alias_name:** `american_to_british`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
american_to_british:
    min_pass_rate: 0.7
```

- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The technician analyzed your samples.|The technician analyzed your samples.|
|What color is this?|What colour is this?|

</div>