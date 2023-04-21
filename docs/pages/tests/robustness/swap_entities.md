
<div class="h3-box" markdown="1">

## Swap Entities

This test shuffles the labeled entities in the input to test the models robustness.

**alias_name:** `swap_entities`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
swap_entities:
    min_pass_rate: 0.8
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|I love Paris.|I love Istanbul.|
|Jack is sick today.|Adam is sick today.|

</div>