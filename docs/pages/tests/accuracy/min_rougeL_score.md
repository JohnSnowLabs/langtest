
<div class="h3-box" markdown="1">

## Min RougeL Score

This test uses the "rouge_score" from evaluate library. This test uses rougeL result of "rouge_score". Test is passed if the score is higher than the configured min score.

**alias_name:** `min_rougeL_score`

**supported tasks:** `question_answering`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_rougeL_score:
      min_score: 0.8
```
- **min_score (float):** Minimum score to pass the test.

<!-- #### Examples -->

</div>