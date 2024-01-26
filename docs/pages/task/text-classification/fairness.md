
<div class="h3-box" markdown="1">

## Fairness

Fairness testing is a critical aspect of evaluating the performance of a machine learning model, especially when the model has potential implications for specific groups of people. Fairness testing aims to ensure that the model is not biased towards or against any particular group and that it produces unbiased results for all groups.

**How it works:**

- We categorize the input data into three groups: "male," "female," and "unknown."
- The model processes *original* sentence from the categorized input data, producing an *actual_result* for each sample.
- The *expected_result* in fairness is the ground truth that we get from the dataset.
- During evaluation, the predicted labels in the expected and actual results are compared for each group to assess the model's performance.

</div>