
## Min Gender F1 Score

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test evaluates the model for each gender seperately. The f1 score for each gender is calculated and test is passed if they are higher than config.

**alias_name:** `min_gender_f1_score`

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


</div></div>