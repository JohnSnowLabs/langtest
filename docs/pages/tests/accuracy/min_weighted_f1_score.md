
<div class="h3-box" markdown="1">

## Min Weighted-F1 Score

This test checks the weighted-f1 score. Test is passed if the weighted-f1 score is higher than the configured min score.

**alias_name:** `min_weighted_f1_score`

**supported tasks:** `ner`, `text-classification`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_weighted_f1_score:
      min_score: 0.8
```

- **min_score (float):** Minimum pass rate to pass the test.

<!-- #### Examples -->

</div>