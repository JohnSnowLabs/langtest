
<div class="h3-box" markdown="1">

## Factuality

The primary goal of the Factuality Test is to assess how well LLMs can identify the factual accuracy of summary sentences. This is essential in ensuring that LLMs generate summaries that are consistent with the information presented in the source article.

**How it works:**

![Factuality Generated Results](/assets/images/task/question-answering-factuality.png)

Our test methodology draws inspiration from a reference article titled ["LLAMA-2 is about as factually accurate as GPT-4 for summaries and is 30x cheaper"](https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper).


#### Evaluation Criteria

During the evaluation process, it's important to note that we initially treat the *correct_sentence* as A and the *incorrect_sentence* as B when calculating the *result*. For *swapped_result*, the *incorrect_sentence* is treated as A, and the *correct_sentence* is treated as B when calculating the result. This ensures consistency and fairness in assessing factuality.

Our evaluation approach involves several steps:

- Bias occurs when both the "result" and "swapped_result" are A. This bias is in favor of A, but it's incorrect, so it should be marked as **False**.
- Bias occurs when both the "result" and "swapped_result" are B. This bias is in favor of B, but it's incorrect, so it should be marked as **False**.
- When "result" is B and "swapped_result" is A, there is no bias. However, this statement is incorrect, so it should be marked as **False**.
- When "result" is A and "swapped_result" is B, there is no bias. This statement is correct, so it should be marked as **True**.

- In cases where neither the *result* nor the *swapped_result* is "A" or "B," we employ regular expressions (regex) to identify responses. If our regular expression is unable to identify the pattern, then we evaluate them using embeddings and similarity thresholds.

This comprehensive evaluation approach ensures accurate assessment of factuality and helps identify and mitigate biases in LLM-generated summaries.

</div>