
## Min Gender Representation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of genders. The test checks the count or proportion of genders depending on the alias_name.

**alias_name:** `min_religion_name_representation_count` or `min_religion_name_representation_proportion`


#### Config
```yaml
min_religion_name_representation_count:
    min_count: 
        christian: 10
        muslim: 5
        hindu: 8
        parsi: 40
        sikh: 10
```

```yaml
min_religion_name_representation_proportion:
    min_proportion: 
        muslim: 0.4
        hindu: 0.5
                
```
- **min_count:** Minimum count to pass the test.
- **min_proportion:** Minimum proportion to pass the test.

<!-- #### Examples -->
