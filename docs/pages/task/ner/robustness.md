
<div class="h3-box" markdown="1">

## Robustness

Robustness testing aims to evaluate the ability of a model to maintain consistent performance when faced with various perturbations or modifications in the input data.

**How it works:**

{:.table3}
| test_type         | original                    | test_case                   | expected_result | actual_result |  pass  |
|-------------------|--------------------------------------|--------------------------------------|----------------------------------------------|-----------------|---------------|-------|
| 	add_ocr_typo | " He ended the World Cup on the wrong note , " Coste said . | " He ended <span style="color:red">t^e woiUd</span> Cup on <span style="color:red">t^e v/rong notc</span> , " Coste said . | World Cup: MISC,<br> Coste: PER	      | woiUd Cup: MISC,<br> Coste: PER	   |<span style="color:green">True</span>  |
| uppercase        | Despite winning the Asian Games title two years ago , Uzbekistan are in the finals as outsiders .	             | <span style="color:red">DESPITE WINNING THE ASIAN GAMES TITLE TWO YEARS AGO , UZBEKISTAN ARE IN THE FINALS AS OUTSIDERS .</span>             | 	Asian Games: MISC,<br> Uzbekistan: LOC	       | ASIAN: MISC,<br> UZBEKISTAN: LOC    |<span style="color:red">False</span>     |

- Perturbations, such as lowercase, uppercase, typos, etc., are introduced to the *original* text, resulting in a perturbed *test_case*.
-  It's important to note that when we add perturbations to the original text, we also track the span of the words that are perturbed.  This allows us to determine the indices of those words, simplifying the process of realigning the original text with the perturbed text.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 
- During evaluation, the predicted entities in the expected and actual results are compared to assess the model's performance

</div>