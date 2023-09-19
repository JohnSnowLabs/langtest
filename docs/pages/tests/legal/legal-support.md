
<div class="h3-box" markdown="1">

## Legal-Support

The LegalSupport dataset evaluates fine-grained reverse entailment. Each sample consists of a text passage making a legal claim, and two case summaries. Each summary describes a legal conclusion reached by a different court. The task is to determine which case (i.e. legal conclusion) most forcefully and directly supports the legal claim in the passage. The construction of this benchmark leverages annotations derived from a legal taxonomy expliciting different levels of entailment (e.g. "directly supports" vs "indirectly supports"). As such, the benchmark tests a model's ability to reason regarding the strength of support a particular case summary provides.


**alias_name:** `legal-support`

</div><div class="h3-box" markdown="1">

#### Config
```yaml
legal-support:
    min_pass_rate: 0.7
```
- **min_pass_rate (float):** Minimum pass rate to pass the test.

</div><div class="h3-box" markdown="1">


</div>
