
<div class="h3-box" markdown="1">


## Robustness

The main objective of model robustness tests is to assess a model's capacity to sustain consistent output when exposed to perturbations in the data it predicts.

**How it works:**

![Robustness Testing In Question Answering](/assets/images/task/question-answering-robustness.png)

- Introducing perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*.
- The model processes both the original and perturbed inputs, resulting in *expected_result* and *actual_result* respectively. 

> For our evaluation metric, we employ a two-layer method where the comparison between the expected_result and *actual_result* is conducted:

![Robustness Evaluation Metric For Question Answering](/assets/images/task/question-answering-robustness-evaluation.png)

- Layer 1: Checking if the *expected_result* and *actual_result* are the same by directly comparing them.
However, this approach encounters challenges when weak LLMs fail to provide answers in alignment with the given prompt, leading to inaccuracies.

- layer 2: If the initial evaluation using the direct comparison approach proves inadequate, we move to Layer 2. we provide three alternative options for evaluation: [String distance](/docs/pages/misc/string_distance), [Embedding distance](/docs/pages/misc/embedding_distance), or utilizing [LLM Eval](/docs/pages/misc/llm_eval).
> This dual-layered approach enhances the robustness of our evaluation metric, allowing for adaptability in scenarios where direct comparisons may fall short.

For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.

## Bias

The primary goal of  bias tests is to explore how replacing documents with different genders, ethnicities, religions, or countries impacts the model’s predictions compared to the original training set.

**How it works:**

![Bias Testing In Question Answering](/assets/images/task/question-answering-bias.png)

To evaluate Question Answering Models for bias, we use the curated dataset: `BoolQ` and split : `bias`

- No perturbations are added to the inputs, as the dataset already contains those perturbations.
- The model processes these original inputs and perturbed inputs, producing an *expected_result* and *actual_result* respectively. 
- The evaluation process employs the same approach used in robustness.

## Accuracy
 
Accuracy testing is vital for evaluating a NLP model’s performance. It gauges the model’s ability to predict outcomes on an unseen test dataset by comparing predicted and ground truth.

**How it works:**

- The model processes *original_context* and *original_question*, producing an *actual_result*.
- The *expected_result* in accuracy is the ground truth.
- In the evaluation process, we compare the *expected_result* and *actual_result* based on the tests within the accuracy category.

## Fairness

The primary goal of fairness testing is to assess the performance of a NLP model without bias, particularly in contexts with implications for specific groups. This testing ensures that the model does not favor or discriminate against any group, aiming for unbiased results across all groups.

**How it works:**

- We categorize the input data into three groups: "male," "female," and "unknown."
- The model processes *original_context* and *original_question* from the categorized input data, producing an *actual_result* for each sample.
- The *expected_result* in fairness is the ground truth that we get from the dataset.
- In the evaluation process, we compare the *expected_result* and *actual_result* based on the tests within the bias category.

## representation

The goal of representation testing is to assess whether a dataset accurately represents a specific population or if biases within it could adversely affect the results of any analysis.

How it works:

- From the dataset, it extracts the *original_question* and *original_context*.
- Subsequently, it employs a classifier or dictionary to calculate representation_proportion and representation_count based on the applied test.


</div>