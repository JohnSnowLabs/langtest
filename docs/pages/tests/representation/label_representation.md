
## Min Label Representation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks the data regarding the sample counts of labels. The test checks the count or proportion of labelsdepending on the alias_name and config.

**alias_name:** `min_label_representation_count` or `min_label_representation_proportion`


#### Config
```yaml
min_label_representation_count:
    min_count: 
        positive: 10
        negative: 10
```

```yaml
min_label_representation_proportion:
    min_proportion: 
        O: 0.6
        LOC: 0.7
                
```
- **min_count:** Minimum count to pass the test.
- **min_proportion:** Minimum proportion to pass the test.

<!-- #### Examples -->
