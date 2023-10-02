
<div class="h3-box" markdown="1">


## Toxicity

The Evaluating Model's Sensitivity to Toxicity Test focuses on assessing a model's responsiveness to toxicity and its ability to handle inputs containing toxic words. The primary objective is to determine whether the model can effectively detect and respond to toxic language. This evaluation consists of two key stages: first, introducing toxic words into the input text, and second, observing how the model behaves or how sensitive it is when presented with input containing these toxic words.

**alias_name:** `toxicity`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
  sensitivity:
    toxicity:
      min_pass_rate: 0.70
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">


</div>