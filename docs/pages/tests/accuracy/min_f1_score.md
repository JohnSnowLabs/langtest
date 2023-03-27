
## Min F1 Score

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the f1 score for each label. Test is passed if the f1 score is higher than the configured min score.

**alias_name:** `min_f1_score`

</div><div class="h3-box" markdown="1">

### Config
```yaml
min_f1_score:
      min_score: 0.8
```
```yaml
min_f1_score:
      min_score:
        O: 0.75
        PER: 0.65
        LOC: 0.90
```
**min_pass_rate:** Minimum pass rate to pass the test.

<!-- ### Examples -->

</div></div>