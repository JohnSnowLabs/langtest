
<div class="h3-box" markdown="1">



## Order Bias

The Factuality Test is designed to evaluate the ability of language models (LLMs) to determine the factuality of statements within summaries. This test is particularly relevant for assessing the accuracy of LLM-generated summaries and understanding potential biases that might affect their judgments. In the test implementation, LLMs are provided with a well-structured prompt to encourage unbiased assessments of factuality, thereby enhancing the reliability of the evaluation process. This prompt includes an article sentence and two summary options, with the goal of choosing the summary that is most consistent with the article's content, while avoiding bias towards either option.

**alias_name:** `order_bias`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
  factuality:
    order_bias:
      min_pass_rate: 0.70
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

#### Example

{:.table3}
| article_sentence | correct_sentence | incorrect_sentence | result | swapped_result | pass |
|-------------------|------------------|---------------------|--------|-----------------|------|
| the abc have reported that those who receive centrelink payments made up half of radio rental's income last year. | those who receive centrelink payments made up half of radio rental's income last year. | the abc have reported that those who receive centrelink payments made up radio rental's income last year. | A | B | True |


</div>