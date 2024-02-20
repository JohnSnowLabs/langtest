
<div class="h3-box" markdown="1">

## Titlecase

This test checks if the NLP model can handle input text that is in titlecase format, where the first letter of each word is capitalized.

**alias_name:** `titlecase`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
titlecase:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `titlecase` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The quick brown fox jumps over the lazy dog.|<span style="color:red">T</span>he <span style="color:red">Q</span>uick <span style="color:red">B</span>rown <span style="color:red">F</span>ox <span style="color:red">J</span>umps <span style="color:red">O</span>ver <span style="color:red">T</span>he <span style="color:red">L</span>azy <span style="color:red">D</span>og.|
|I LIKE TO SHOUT.|<span style="color:red">I</span> <span style="color:red">L</span>ike <span style="color:red">T</span>o <span style="color:red">S</span>hout.|

</div>
