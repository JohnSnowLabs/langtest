
<div class="h3-box" markdown="1">

## Min Bleu Score

This test uses the "bleu_score" from evaluate library. Test is passed if the score is higher than the configured min score.

**alias_name:** `min_bleu_score`

**supported tasks:** `question_answering`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_bleu_score:
      min_score: 0.8
```
- **min_score (float):** Minimum score to pass the test.

<!-- #### Examples -->

</div>