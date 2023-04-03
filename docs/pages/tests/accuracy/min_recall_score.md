
## Min Recall Score

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the recall score for each label. Test is passed if the recall score is higher than the configured min score.

**alias_name:** `min_recall_score`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_recall_score:
      min_score: 0.5
```
```yaml
min_recall_score:
      min_score:
        O: 0.75
        PER: 0.65
        LOC: 0.90
```
- **min_score (dict or float):** Minimum pass rate to pass the test.

<!-- #### Examples -->

</div></div>