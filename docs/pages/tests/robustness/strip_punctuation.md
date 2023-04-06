
## Strip Punctuation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle sentences without punctuations at the end.

**alias_name:** `strip_punctuation`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
strip_punctuation:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog|
|Good morning!|Good morning|


</div></div>