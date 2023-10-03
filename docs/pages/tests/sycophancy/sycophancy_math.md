
<div class="h3-box" markdown="1">


## Sycophancy Math

Our process revolves around the creation of synthetic mathematical data, aimed at gauging the model's alignment with user prompts, regardless of the correctness of the mathematical assertions. By crafting these synthetic inputs, we seek to explore how effectively the model can interpret and respond to mathematical opinions and assertions, shedding light on its ability to comprehend and engage in mathematical discourse, irrespective of the factual accuracy of the statements. Furthermore, through interventions such as swapping 'A' to disagree and 'B' to agree, we introduce an additional layer of complexity, allowing us to scrutinize how the model adapts to diverse mathematical options, thus shedding light on its adaptability and reasoning within the realm of mathematical discourse.

**alias_name:** `sycophancy_math`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
  sycophancy:
    sycophancy_math:
      min_pass_rate: 0.70
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">

</div>