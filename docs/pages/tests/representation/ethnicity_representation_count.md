
<div class="h3-box" markdown="1">

## Min Ethnicity Representation Count

This test checks the data regarding the sample counts of ethnicities.

**alias_name:** `min_ethnicity_name_representation_count`

<i class="fa fa-info-circle"></i>
<em>This data was curated using 2021 US census survey data. To apply this test appropriately in other contexts, please adapt the [data dictionaries](https://github.com/JohnSnowLabs/langtest/blob/main/langtest/transform/constants.py).</em>

#### Config
```yaml
min_ethnicity_name_representation_count:
    min_count: 
        white: 50
        black: 10
        asian: 40
        hispanic: 30           
```
- **min_count (int):** Minimum count to pass the test.

<!-- #### Examples -->

</div>