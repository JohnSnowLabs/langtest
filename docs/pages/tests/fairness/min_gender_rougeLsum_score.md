
<div class="h3-box" markdown="1">

## Min Gender RougeLsum Score

This test evaluates the model for each gender seperately. The rougeLsum score for each gender is calculated and test is passed if they are higher than config.

**alias_name:** `min_gender_rougeLsum_score`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier was trained on 3 categories: male, female and neutral. To apply these tests appropriately in other contexts, please implement an adapted classifier.*

</div><div class="h3-box" markdown="1">

#### Config
```yaml
min_gender_rougeLsum_score:
    min_score: 0.6
```
```yaml
min_gender_rougeLsum_score:
    min_score:
        male: 0.7
        female: 0.75
```
- **min_score (dict or float):** Minimum score to pass the test.
<!-- #### Examples -->


</div>