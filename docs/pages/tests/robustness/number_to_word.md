<div class="h3-box" markdown="1">

## Number To Word

This test provides a convenient way to convert numerical values in text into their equivalent words. It is particularly useful for applications that require handling or processing textual data containing numbers.

**alias_name:** `number_to_word`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
number_to_word:
    min_pass_rate: 0.8
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

| Original                                                                                                | Test Case                                                                                                       |
|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| I live in London, United Kingdom since 2019                                                             | I live in London, United Kingdom since two thousand and nineteen                                                |
| I can't move to the USA because they have an average of 100 tornadoes a year, and I'm terrified of them | I can't move to the USA because they have an average of one hundred tornadoes a year, and I'm terrified of them |

</div>
