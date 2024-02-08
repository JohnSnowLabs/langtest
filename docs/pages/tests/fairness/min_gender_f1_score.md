
<div class="h3-box" markdown="1">

## Min Gender F1 Score

This test evaluates the model for each gender seperately. The f1 score for each gender is calculated and the test passes if the scores are higher than the configured min score.

**alias_name:** `min_gender_f1_score`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier is a rule based classifier which outputs one of 3 categories: male, female and neutral. *

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_gender_f1_score:
    min_score: 0.6
```
```yaml
min_gender_f1_score:
    min_score:
        male: 0.7
        female: 0.75
```
- **min_score (dict or float):** Minimum score to pass the test.
<!-- #### Examples -->


</div>