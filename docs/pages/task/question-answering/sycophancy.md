
<div class="h3-box" markdown="1">

## Sycophancy

The primary goal of addressing sycophancy in language models is to mitigate undesirable behaviors where models tailor their responses to align with a human user’s view, even when that view is not objectively correct.

**How it works:**


![Sycophancy Generated Results](/assets/images/task/question-answering-sycophancy.png)

- The process begins by introducing perturbations, specifically by adding a human prompt at the beginning of the *original_question*, resulting in the creation of a *perturbed_question*.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 

#### Evaluation Criteria 

We gauge the model's performance by comparing the expected result with what it actually produces using the [LLM Eval](/docs/pages/misc/llm_eval) metric.

*Note: If the user wants to consider the ground truth (which can be specified through the config), we conduct a parallel assessment. This involves comparing the ground truth with both the expected result and the actual result. This helps us determine if the model's response meets the evaluation criteria.*

</div>