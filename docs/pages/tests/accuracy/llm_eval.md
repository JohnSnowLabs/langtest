
<div class="h3-box" markdown="1">

## LLM Eval

we employ a more robust Language Model (LLM) to evaluate the model’s response. Test is passed if the score is higher than the configured min score.

**alias_name:** `llm_eval`

**supported tasks:** `question_answering`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
llm_eval:
    hub: openai
    min_score: 0.75
    model: gpt-3.5-turbo-instruct
```
- **min_score (float):** Minimum score to pass the test.
- **model (string):** LLM model use to evaluate the model reponse.
- **hub (string):**   Hub (library) for loading model from public models hub or from path 

<!-- #### Examples -->

</div>