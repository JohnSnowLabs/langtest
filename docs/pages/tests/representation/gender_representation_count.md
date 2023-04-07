
## Min Gender Representation Count

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of genders. 

**alias_name:** `min_gender_representation_count`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier was trained on 3 categories: male, female and neutral. To apply these tests appropriately in other contexts, please implement an adapted classifier.*

#### Config
```yaml
min_gender_representation_count:
    min_count: 
        male: 20
        female: 30
```

- **min_count (int):** Minimum count to pass the test.

<!-- #### Examples -->
