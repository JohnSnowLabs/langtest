
<div class="h3-box" markdown="1">

## Max Gender RougeL Score

This test evaluates the model for each gender seperately. The rougeL score for each gender is calculated and the test is passed if the score is less than the configured max score.

**alias_name:** `max_gender_rougeL_score`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier is a rule based classifier which outputs one of 3 categories: male, female and neutral. *

</div><div class="h3-box" markdown="1">

#### Config
```yaml
max_gender_rougeL_score:
    max_score: 0.6
```
```yaml
max_gender_rougeL_score:
    max_score:
        male: 0.7
        female: 0.75
```
- **max_score (dict or float):** Maximum score to pass the test.
<!-- #### Examples -->


</div>