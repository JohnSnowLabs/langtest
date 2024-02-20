
<div class="h3-box" markdown="1">

## Max Gender LLM Eval

This test evaluates the model for each gender seperately. we employ a more robust Language Model (LLM) to evaluate the model’s response. Test is passed if the score is less than the configured max score.

**alias_name:** `max_gender_llm_eval`

<i class="fa fa-info-circle"></i>
*The underlying gender classifier is a rule based classifier which outputs one of 3 categories: male, female and neutral. *

</div><div class="h3-box" markdown="1">

#### Config
```yaml
max_gender_llm_eval:
    hub: openai
    model: gpt-3.5-turbo-instruct
    max_score: 0.6
    
```
```yaml
max_gender_llm_eval:
    hub: openai
    model: gpt-3.5-turbo-instruct
    max_score:
        male: 0.7
        female: 0.75
```

- **model (string):** LLM model use to evaluate the model reponse.
- **hub (string):** Hub (library) for loading model from public models hub or from path
- **max_score (dict or float):** Maximum score to pass the test.
<!-- #### Examples -->


</div>