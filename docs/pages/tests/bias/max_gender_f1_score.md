
## Max Gender F1 Score

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test evaluates the model for each gender seperately. The f1 score for each gender is calculated and test is passed if they are smaller than config.
**alias_name:** `min_gender_f1_score`

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


</div></div>