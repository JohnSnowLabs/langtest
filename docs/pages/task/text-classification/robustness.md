
<div class="h3-box" markdown="1">

## Robustness

Robustness testing aims to evaluate the ability of a model to maintain consistent performance when faced with various perturbations or modifications in the input data.

**How it works:**

{:.table3}
| test_type         | original                    | test_case                   | expected_result | actual_result |  pass  |
|-------------------|--------------------------------------|--------------------------------------|----------------------------------------------|-----------------|---------------|-------|
| 	add_ocr_typo | director rob marshall went out gunning to make a great one . | diredor rob marshall went <span style="color:red">o^ut</span> gunning <span style="color:red">t^o makc</span> a <span style="color:red">grcat o^ne</span> .	      | POSITIVE	   | NEGATIVE  |<span style="color:red">False</span>  |
| uppercase        | an amusing , breezily apolitical documentary about life on the campaign trail .	             | <span style="color:red">AN AMUSING , BREEZILY APOLITICAL DOCUMENTARY ABOUT LIFE ON THE CAMPAIGN TRAIL .</span>             | POSITIVE	       | POSITIVE   | <span style="color:green">True</span>     |

- Perturbations, such as lowercase, uppercase, typos, etc., are introduced to the *original* text, resulting in a perturbed *test_case*.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 
- During evaluation, the predicted labels in the expected and actual results are compared to assess the model's performance

</div>