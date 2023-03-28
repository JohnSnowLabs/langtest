
## Lowercase

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text that is in all lowercase. Like the uppercase test, this is important to ensure that your NLP model can handle input text in any case.

**alias_name:** `lowercase`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
lowercase:
    min_pass_rate: 0.8
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

#### Examples

{:.table2}
|Original|Testcase|
|-|
|The quick brown fox jumps over the lazy dog.|the quick brown fox jumps over the lazy dog.|
|I AM VERY QUIET.|i am very quiet.|


</div></div>