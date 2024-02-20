
<div class="h3-box" markdown="1">

## Grammar

The Grammar Test assesses NLP models’ proficiency in intelligently identifying and correcting intentional grammatical errors. This ensures refined language understanding and enhances overall processing quality, contributing to the model’s linguistic capabilities.

**How it works:**

{:.table3}
| test_type         | original                    | test_case                   | expected_result | actual_result |  pass  |
|-------------------|--------------------------------------|--------------------------------------|----------------------------------------------|-----------------|---------------|-------|
| 	paraphrase | This is one of those movies that showcases a great actor's talent and also conveys a great story. It is one of Stewart's greatest movies. Barring a few historic errors it also does an excellent job of telling the story of the "Spirit of St. Louis". | <span style="color:red">This film is one of those movies that showcases a great actor and also conveys a great story. Stewart considers it one of his greatest accomplishments. Despite a few historical mistakes, it still manages to tell the story of the "Spirit of St. Louis".</span>	      | POSITIVE	   | POSITIVE  |<span style="color:green">True</span>  |
| paraphrase        | I thoroughly enjoyed Manna from Heaven. The hopes and dreams and perspectives of each of the characters is endearing and we, the audience, get to know each and every one of them, warts and all. And the ending was a great, wonderful and uplifting surprise! Thanks for the experience; I'll be looking forward to more.	             | <span style="color:red">Manna from Heaven was a wonderful movie, with all the cute and unexpected things in life. Each character has such a unique and heartfelt personality, and the ending is a surprise that I couldn't have asked for in the past.</span>             | POSITIVE	       | POSITIVE   | <span style="color:green">True</span>     |

- During the perturbation process, we paraphrase the *original* text, resulting in a perturbed *test_case*.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively.
- During evaluation, the predicted labels in the expected and actual results are compared to assess the model's performance.

</div>