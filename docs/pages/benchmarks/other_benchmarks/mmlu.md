---
layout: docs
header: true
seotitle: MMLU Benchmark | LangTest | John Snow Labs
title: MMLU
key: benchmarks-mmlu
permalink: /docs/pages/benchmarks/other_benchmarks/mmlu/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/mmlu_dataset.ipynb)

**Source:** [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)

The MMLU dataset is a collection of questions and answers that cover many subjects across various domains, such as STEM, humanities, social sciences, and more. The dataset is designed to measure the performance of language understanding models on a wide range of tasks, such as elementary mathematics, US history, computer science, law, and more. Each sample has a question, and 4 choices and one of them is correct. The dataset can be used to evaluate the models’ ability to reason, recall facts, and apply knowledge to different scenarios.

You can see which subsets and splits are available below.

{:.table2}
| Split       | Details                                                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **test**    | Test set from the MMLU dataset which covers 57 tasks including elementary mathematics, US history, computer science, law, and more. We took 50 samples from each task in the test set.                             |
| **test-tiny** | Truncated version of the test set from the MMLU dataset which covers 57 tasks including elementary mathematics, US history, computer science, law, and more. We took 10 samples from each task in the test-tiny set. |
| **clinical**    | Curated version of the MMLU dataset which contains the clinical subsets (college_biology, college_medicine, medical_genetics, human_aging, professional_medicine, nutrition).                                       |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    |  original_question                  |  perturbed_question                     |  options      | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|---------------|-------------------------------|-------------------------------|-------|
| robustness | add_speech_to_text_typo | Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. | Find the degree for the given field extension Kew(sqrt(2), sqrt(3), sqrt(18)) over Q.|<br>A. 0<br>B. 4<br>C. 2<br>D. 6 | B. 4 | B. 4  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Benchmarks

{:.table3}
| Models                         | uppercase | lowercase | titlecase	 | add_typo | dyslexia_word_swap | add_abbreviation	 | add_slangs | add_speech_to_text_typo	 | add_ocr_typo | adjective_synonym_swap |
|:-------------------------------:|:---------:|:---------:|:---------:|:--------:|:-------------------:|:----------------:|:----------:|:-----------------------:|:------------:|:-----------------------:|
| j2-jumbo-instruct               |    74%    |    79%    |    81%    |    88%   |         79%         |        73%       |    77%     |           89%           |      75%      |           77%           |
| j2-grande-instruct              |    66%    |    73%    |    72%    |    87%   |         72%         |        72%       |    71%     |           90%           |      69%      |           75%           |
| gpt-3.5-turbo-instruct          |    91%    |    93%    |    93%    |    94%   |         76%         |        83%       |    75%     |           94%           |      88%      |           79%           |
| mistralai/Mistral-7B-Instruct-v0.1 |    72%    |    77%    |    76%    |    93%   |         89%         |        87%       |    88%     |           88%           |      80%      |           88%           |

> Minimum pas rate: 75%

**Dataset info:**
- Split: clinical
- Records: 992
- Testcases: 9048
- Evaluatuion: GPT-3.5-Turbo-Instruct

</div>

![MMLU Clinical Benchmark](/assets/images/benchmarks/mmlu.png)