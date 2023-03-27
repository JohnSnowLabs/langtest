
## AddTypo

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text with typos. A typo frequenct dictionary is used to apply most common typos to the input data.

**alias_name:** `add_typo`

</div><div class="h3-box" markdown="1">

### Config
```yaml
add_context:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

### Examples

{:.table2}
|Original|Testcase|
|-|
|The quick brown fox jumps over the lazy dog.|The wuick brown fox jumps over the fazy dog.|
|Good morning|Good morninh|


</div></div>