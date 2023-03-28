
## Add Punctuation

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text with sentences with a punctuation at the end. The added punctuation is randomly chosen from the list `['!', '?', ',', '.', '-', ':', ';']`. If there already is a punctuation it is not changed.

**alias_name:** `add_contraction`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_context:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Testcase|
|-|
|The quick brown fox jumps over the lazy dog|The quick brown fox jumps over the lazy dog.|
|Good morning|Good morning!|
|I like football.|I like football.|


</div></div>