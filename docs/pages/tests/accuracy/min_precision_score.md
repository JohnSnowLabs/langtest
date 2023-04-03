
## Min Precision Score

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the precision score for each label. Test is passed if the precision score is higher than the configured min score.

**alias_name:** `min_precision_score`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_precision_score:
      min_score: 0.8
```
```yaml
min_precision_score:
      min_score:
        O: 0.75
        PER: 0.65
        LOC: 0.90
```
- **min_score (dict or float):** Minimum pass rate to pass the test.

<!-- #### Examples -->

</div></div>