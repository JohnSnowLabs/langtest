
<div class="h3-box" markdown="1">

## Min Gender Representation Proportion

This test checks the data regarding the sample proportions of genders.

**alias_name:** `min_gender_representation_proportion`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier was trained on 3 categories: male, female and neutral. To apply these tests appropriately in other contexts, please implement an adapted classifier.*

#### Config
```yaml
min_gender_representation_count:
    min_count: 
        male: 0.2
        female: 0.3
```

- **min_proportion (float):** Minimum proportion to pass the test.

<!-- #### Examples -->

</div>