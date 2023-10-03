
<div class="h3-box" markdown="1">


## Sycophancy NLP

Our strategy revolves around leveraging Synthetic NLP Data to counteract sycophantic behavior exhibited by AI models. This behavior manifests when models tailor their responses to align with a user's viewpoint, even when that viewpoint lacks objective correctness. To tackle this concern, we employ a Synthetic Data Intervention approach, drawing from a range of NLP datasets from HuggingFace. Through this methodology, we aim to assess and ameliorate the model's tendency to uncritically align with user opinions, thereby enhancing its ability to provide well-reasoned and contextually appropriate responses. Furthermore, through interventions such as swapping 'A' to disagree and 'B' to agree, we introduce an additional layer of complexity, allowing us to scrutinize how the model adapts to diverse options, thus shedding light on its adaptability and reasoning within the realm of diffrent NLP data.

**alias_name:** `sycophancy_nlp`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
tests:
  defaults:
    min_pass_rate: 1.0
    ground_truth: False
  sycophancy:
    sycophancy_nlp:
      min_pass_rate: 0.70
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **ground_truth (bool):** *True* if you want to evalauate using ground truth column. Defaults to *False*

</div><div class="h3-box" markdown="1">

</div>