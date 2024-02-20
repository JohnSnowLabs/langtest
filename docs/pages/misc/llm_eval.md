---
layout: docs
header: true
seotitle: LLM Eval | LangTest | John Snow Labs
title: LLM Eval
key: misc-examples
permalink: /docs/pages/misc/llm_eval
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

we employ a more robust Language Model (LLM) to evaluate the model’s response.
How it operates in LangTest for robustness testing:

- The evaluation process is conducted on provided data, by assessing the *original_question* and *expected results*(ground truth), as well as the *perturbed question* and *actual results*.
- The outcome of the evaluation process determines whether the *actual results* aligns with the *expected results* (ground truth).

</div></div><div class="h3-box" markdown="1">

### Configuration Structure

To configure string distance metrics, you can use a YAML configuration file. The configuration structure includes:

- `model_parameters` specifying model-related parameters.
- `evaluation` setting the evaluation `metric`, `model`, and `hub`.
- `tests` defining different test scenarios and their `min_pass_rate`.

Here's an example of the configuration structure:

```yaml
model_parameters:
  temperature: 0.2
  max_tokens: 64

evaluation:
  metric: llm_eval
  model: gpt-3.5-turbo-instruct
  hub: openai

tests:
  defaults:
    min_pass_rate: 1.0

  robustness:
    add_typo:
      min_pass_rate: 0.70
    lowercase:
      min_pass_rate: 0.70
```