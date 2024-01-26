
<div class="h3-box" markdown="1">

## Bias

Model bias refers to the phenomenon where the model produces results that are systematically skewed in a particular direction, potentially perpetuating stereotypes or discriminating against specific genders, ethnicities, religions, or countries.

**How it works:**

- Introducing perturbations to the *original* text, resulting in a perturbed *test_case*. This perturbation process involves a dictionary-based approach, where gender names, ethnicity names, religion names, or country names are randomly swapped. Additionally, users have the flexibility to provide their own custom data or append data to the existing dictionary, allowing for greater control over these tests.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 
- During evaluation, the predicted labels in the expected and actual results are compared to assess the model's performance.

</div>