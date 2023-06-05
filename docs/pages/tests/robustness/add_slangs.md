
<div class="h3-box" markdown="1">

## Add Slangs

This test involves substituting certain words (specifically nouns, adjectives, and adverbs) in the original text with their corresponding slang terms. The purpose is to assess the NLP model's ability to handle input text that includes slang language.

**alias_name:** `add_slangs`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
add_slangs:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|It was totally excellent but useless bet.|It was totes grand but cruddy flutter.|
|Obviously, money are a great stimulus but people might go to crazy about it.|Obvs, spondulicks are a nang stimulus but peeps might go to radio rental about it.|

</div>