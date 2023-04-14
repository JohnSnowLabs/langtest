
<div class="h3-box" markdown="1">

## Titlecase

This test checks if the NLP model can handle input text that is in titlecase format, where the first letter of each word is capitalized.

**alias_name:** `titlecase`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
titlecase:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|The Quick Brown Fox Jumps Over the Lazy Dog.|
|I LIKE TO SHOUT.|I Like To Shout.|

</div>