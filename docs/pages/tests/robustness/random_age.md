
<div class="h3-box" markdown="1">

## Randomize Age

This test checks if the NLP model can handle age differences. The test replaces age statements like "x years old" with x ± random_amount. The value is set to 1 if its smaller than 0.

**alias_name:** `randomize_age`

<i class="fa fa-info-circle"></i>
<em>To test QA models, we are using QAEval from Langchain where we need to use the model itself or other ML model for evaluation, which can make mistakes.</em>

</div><div class="h3-box" markdown="1">

#### Config
```yaml
randomize_age:
    min_pass_rate: 0.65
    prob: 1.0 # Defaults to 1.0, which means all statements will be transformed.
    parameters:
      random_amount: 5 # 
      count: 1 # Defaults to 1
```
<i class="fa fa-info-circle"></i>
<em>You can adjust the level of transformation in the sentence by using the "`prob`" parameter, which controls the proportion of statements to be changed during `randomize_age` test.</em>

- **min_pass_rate (float):** Minimum pass rate to pass the test.
- **random_amount (int):** Range of random value to be added/substracted from existing age value.
- **prob (float):** Controls the proportion of statements to be changed.
- **count (int):** Number of variations of sentence to be constructed.

</div><div class="h3-box" markdown="1">

#### Examples

{:.table2}
|Original|Test Case|
|-|
|The baby was 20 days old.|The baby was <span style="color:red">23</span> days old.|
|My grandfather got sick when he was 89 years old.|My grandfather got sick when he was <span style="color:red">80</span> years old.|


</div>
