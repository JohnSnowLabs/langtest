
## Min Country Economic Representation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of countries by economic levels. The test checks the count or proportion of these countries depending on the alias_name.

**alias_name:** `min_religion_name_representation_count` or `min_religion_name_representation_proportion`


#### Config
```yaml
min_country_economic_representation_count:
    min_proportion: 
        high_income: 50
        low_income: 50

```

```yaml
min_country_economic_representation_proportion:
    min_proportion: 
        high_income: 0.6
        low_income: 0.1
```
- **min_count:** Minimum count to pass the test.
- **min_proportion:** Minimum proportion to pass the test.

<!-- #### Examples -->
