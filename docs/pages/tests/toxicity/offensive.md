
<div class="h3-box" markdown="1">

## Offensive

This test checks the toxicity of the completion. Offensive toxicity refers to "abusive speech targeting specific group characteristics, such as ethnic origin, religion, gender, or sexual orientation." Example offensive language, hate speech, cyberbullying, and trolling. This test is more general compared to other type-specific toxicity tests.
This test uses HF evaluate library's 'toxicity' metric. More can be found [here](https://huggingface.co/spaces/evaluate-measurement/toxicity).

**alias_name:** `offensive`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
offensive:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">


</div>
