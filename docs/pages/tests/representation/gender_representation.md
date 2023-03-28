
## Min Gender Representation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of genders. The test checks the count or proportion of genders depending on the alias_name.

**alias_name:** `min_gender_representation_count` or `min_gender_representation_proportion`


#### Config
```yaml
min_gender_representation_count:
    min_count: 
        male: 20
        female: 30
```

```yaml
min_gender_representation_proportion:
    min_proportion: 
        male: 0.33
        female: 0.33
                
```
- **min_count:** Minimum count to pass the test.
- **min_proportion:** Minimum proportion to pass the test.

<!-- #### Examples -->
