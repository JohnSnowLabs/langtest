
<div class="h3-box" markdown="1">


## Robustness

The main objective of model robustness tests is to assess a model's capacity to sustain consistent output when exposed to perturbations in the data it predicts.

**How it works:**


{:.table3}
| test_type         | original                    | test_case                   | expected_result | actual_result |eval_score|  pass  |
|-------------------|--------------------------------------|--------------------------------------|----------------------------------------------|-----------------|---------------|-------|
| 	lowercase | Are you coming back tomorrow? | <span style="color:red">are you coming back tomorrow?</span> | Kommen Sie morgen zurück?	      | kehren Sie morgen zurück?	   | 0.004109  |<span style="color:green">True</span>  |
| uppercase        | Are you ever wrong?	             | <span style="color:red">ARE YOU EVER WRONG?</span>             | 	Haben Sie sich jemals geirrt?	       | IST SIE JEGLICHE WRONG?   | 0.462017   |<span style="color:red">False</span>     |

- Perturbations, such as lowercase, uppercase, typos, etc., are introduced to the *original* text, resulting in a perturbed *test_case*.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 

#### Evaluation Criteria

- The evaluation begins by obtaining embeddings for both the original and perturbed inputs, as well as for the *expected_result* and *actual_result*.
- We then calculate the cosine distances between the embeddings of the original and perturbed inputs, saved as `original_similarities`.
- Similarly, we compute the cosine distances between the embeddings of the *expected_result* and *actual_result*, stored as `translation_similarities`.
- Next, we compare the difference between original_similarities and translation_similarities against a fixed threshold of 0.1.
- If the absolute difference is less than 0.1, indicating close similarity, the test passes.

</div>