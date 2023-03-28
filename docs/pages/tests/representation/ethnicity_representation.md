
## Min Ethnicity Representation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of ethnicities. The test checks the count or proportion of  ethnicities depending on the alias_name and config.

**alias_name:** `min_ethnicity_name_representation_count` or `min_ethnicity_name_representation_proportion`


#### Config
```yaml
min_ethnicity_name_representation_count:
    min_count: 
        white : 50
        black: 10
        asian: 40
        hispanic: 30           
```

```yaml
min_ethnicity_name_representation_proportion:
    min_proportion: 
        white : 0.20
        black: 0.36
                
```
- **min_count:** Minimum count to pass the test.
- **min_proportion:** Minimum proportion to pass the test.

<!-- #### Examples -->
