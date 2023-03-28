
## Add Context

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

{:.h2-select}
This test checks if the NLP model can handle input text with added context, such as a greeting or closing.

**alias_name:** `add_context`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_context:
    min_pass_rate: 0.6
    starting_context: ["Hello,", "Hey"]
    ending_context: ["Bye"]
    strategy: "combined"
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **starting_context (<List[str]>):** Phrases to be added at the start of inputs.
- **ending_context (<List[str]>):** Phrases to be added at the end of inputs.
- **strategy ('start', 'end' or 'combined'):** Which places to add the given phrases.

#### Examples

{:.table2}
|Original|Testcase|
|-|
|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog, bye.|
|I love playing football.|Hello, I love playing football.|


</div></div>