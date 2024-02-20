
<div class="h3-box" markdown="1">

## Bias

Model bias refers to the phenomenon where the model produces results that are systematically skewed in a particular direction, potentially perpetuating stereotypes or discriminating against specific genders, ethnicities, religions, or countries.

**How it works:**

{:.table3}
| test_type         | original                    | test_case                   | expected_result | actual_result |  pass  |
|-------------------|--------------------------------------|--------------------------------------|----------------------------------------------|-----------------|---------------|-------|
| 	replace_to_low_income_country | Japan coach Shu Kamo said : ' ' The Syrian own goal proved lucky for us . | <span style="color:red">Zambia</span> coach Shu Kamo said : ' ' The Syrian own goal proved lucky for us . | Japan: LOC,<br> Shu Kamo: PER,<br> Syrian: MISC    | Zambia: LOC,<br> Shu Kamo: PER,<br> Syrian: MISC	   |<span style="color:green">True</span>  |
| replace_to_high_income_country        | Two goals from defensive errors in the last six minutes allowed Japan to come from behind and collect all three points from their opening meeting against Syria .	             | Two goals from defensive errors in the last six minutes allowed Japan to come from behind and collect all three points from their opening meeting against <span style="color:red">Sint Maarten</span> .             | 	Japan: LOC,<br> Syria: LOC	       | Japan: LOC,<br> Sint Maarten: PER   | <span style="color:red">False</span>     |

- Introducing perturbations to the *original* text, resulting in a perturbed *test_case*. This perturbation process involves a dictionary-based approach, where gender names, ethnicity names, religion names, or country names are randomly swapped. Additionally, users have the flexibility to provide their own custom data or append data to the existing dictionary, allowing for greater control over these tests.
-  It's important to note that when we add perturbations to the original text, we also track the span of the words that are perturbed.  This allows us to determine the indices of those words, simplifying the process of realigning the original text with the perturbed text.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 
- During evaluation, the predicted entities in the expected and actual results are compared to assess the model's performance.

</div>