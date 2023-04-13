
<div class="h3-box" markdown="1">

## Max Gender F1 Score

This test evaluates the model for each gender seperately. The f1 score for each gender is calculated and test is passed if they are smaller than config.

**alias_name:** `min_gender_f1_score`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier was trained on 3 categories: male, female and neutral. To apply these tests appropriately in other contexts, please implement an adapted classifier.*

</div><div class="h3-box" markdown="1">

#### Config
```yaml
max_gender_f1_score:
    max_score: 0.6
```
```yaml
max_gender_f1_score:
    max_score:
        male: 0.7
        female: 0.75
```
- **max_score (dict or float):** Maximum score to pass the test.
<!-- #### Examples -->

</div>