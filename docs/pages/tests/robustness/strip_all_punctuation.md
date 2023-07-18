
<div class="h3-box" markdown="1">

## Strip All Punctuation

This test checks if the NLP model can handle sentences with no punctuations.

**alias_name:** `strip_all_punctuation`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
strip_all_punctuation:
    min_pass_rate: 0.7
    prob: 0.5 # Defaults to 1.0, which means all words will be transformed.
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of words to be changed during `strip_all_punctuation` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **prob (float):** Controls the proportion of words to be changed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|Dutasteride 0.5 mg Capsule Sig : One ( 1 ) Capsule PO once a day.|Dutasteride 0.5 mg Capsule Sig  One ( 1 ) Capsule PO once a day|
|In conclusion , RSDS is a relevant osteoarticular complication in patients receiving either anticalcineurinic drug ( CyA or tacrolimus ) , even under monotherapy or with a low steroid dose.|In conclusion  RSDS is a relevant osteoarticular complication in patients receiving either anticalcineurinic drug ( CyA or tacrolimus )  even under monotherapy or with a low steroid dose|

</div>
