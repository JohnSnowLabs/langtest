
<div class="h3-box" markdown="1">

## Min Rouge1 Score

This test uses the "rouge_score" from evaluate library. This test uses rouge1 result of "rouge_score". Test is passed if the score is higher than the configured min score.

**alias_name:** `min_rouge1_score`

**supported tasks:** `question_answering`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_rouge1_score:
      min_score: 0.8
```
- **min_score (float):** Minimum score to pass the test.

<!-- #### Examples -->

</div>