
## Min Religion Name Representation Count

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of religions.

**alias_name:** `min_religion_name_representation_count`

<i class="fa fa-info-circle"></i>
<em>This data was curated from Kidpaw and JSL data. To apply this test appropriately in other contexts, please adapt the [data dictionaries.](https://github.com/JohnSnowLabs/nlptest/blob/main/nlptest/transform/utils.py)</em>

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

- **min_count (int):** Minimum count to pass the test.

<!-- #### Examples -->
