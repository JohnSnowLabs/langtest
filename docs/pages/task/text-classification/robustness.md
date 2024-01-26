
<div class="h3-box" markdown="1">

## Robustness

Robustness testing aims to evaluate the ability of a model to maintain consistent performance when faced with various perturbations or modifications in the input data.

**How it works:**

- Perturbations, such as capitalization, punctuation, typos, contractions, etc., are introduced to the *original* text, resulting in a perturbed *test_case*.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 
- During evaluation, the predicted labels in the expected and actual results are compared to assess the model's performance

</div>