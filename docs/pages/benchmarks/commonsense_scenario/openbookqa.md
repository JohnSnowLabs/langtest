---
layout: docs
header: true
seotitle: OpenBookQA Benchmark | LangTest | John Snow Labs
title: OpenBookQA
key: benchmarks-openbookqa
permalink: /docs/pages/benchmarks/commonsense_scenario/openbookqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/OpenbookQA_dataset.ipynb)

**Source:** [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/abs/1809.02789)

The OpenBookQA dataset is a collection of multiple-choice questions that require complex reasoning and inference based on general knowledge, similar to an “open-book” exam. The questions are designed to test the ability of natural language processing models to answer questions that go beyond memorizing facts and involve understanding concepts and their relations. The dataset contains 500 questions, each with four answer choices and one correct answer. The questions cover various topics in science, such as biology, chemistry, physics, and astronomy.

You can see which subsets and splits are available below.

{:.table2}
| Split                    | Details                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the OpenBookQA dataset, containing 500 multiple-choice elementary-level science questions |
| **test-tiny** | Truncated version of the test set from the OpenBookQA dataset, containing 50 multiple-choice examples.     |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.

{:.table3}
| category   | test_type        | original_question                            | perturbed_question                        |   options                                                  | expected_result                | actual_result                  | pass   |
|------------|------------------|----------------------------------------------|-------------------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | There is most likely going to be fog around: | There is most likely going 2 b fog around:| A. a marsh<br>B. a tundra<br>C. da plains<br>D. a desert   | A. a marsh | A. a marsh  | True |

> Generated Results for `text-davinci-003` model from `OpenAI`

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Benchmarks

{:.table3}
| Model                               | uppercase  | lowercase  | titlecase | add_typo | dyslexia_word_swap | add_abbreviation | add_slangs | add_speech_to_text_typo | add_ocr_typo | adjective_synonym_swap  | adjective_antonym_swap |
|--------------------------------------|:---------:|:---------:|:---------:|:-------:|:-------------------:|:----------------:|:----------:|:------------------------:|:------------:|:-----------------------:|:------------------------:|
| j2-jumbo-instruct                   |    58%    |    63%    |    60%    |   74%   |         66%         |       58%        |     59%    |           78%            |      51%      |           62%           |           61%            |
| j2-grande-instruct                  |    53%    |    56%    |    58%    |   79%   |         66%         |       63%        |     64%    |           84%            |      54%      |           65%           |           58%            |
| text-davinci-003                    |    83%    |    85%    |    85%    |   90%   |         80%         |       81%        |     67%    |           88%            |      85%      |           71%           |           66%            |
| gpt-3.5-turbo-instruct              |    87%    |    88%    |    85%    |   90%   |         80%         |       79%        |     66%    |           91%            |      79%      |           68%           |           66%            |
| mistralai/Mistral-7B-Instruct-v0.1  |    55%    |    59%    |    54%    |   88%   |         85%         |       81%        |     76%    |           87%            |      72%      |           78%           |           74%            |
| HuggingFaceH4/zephyr-7b-beta         |    52%    |    46%    |    53%    |   83%   |         86%         |       76%        |     74%    |           82%            |      63%      |           80%           |           69%            |
| Intel/neural-chat-7b-v3-1           |    79%    |    88%    |    82%    |   89%   |         86%         |       84%        |     80%    |           86%            |      73%      |           80%           |           71%            |
| gpt-4-1106-preview                   |    94%    |    93%    |    94%    |   94%   |         90%         |       87%        |     72%    |           93%            |      92%      |           75%           |           64%            |

**Dataset info:**

- Split: test
- Records: 500
- Testcases: 4813
- Evaluatuion: GPT-3.5-Turbo-Instruct

</div>

![OpenBookQA Benchmark](/assets/images/benchmarks/openbookqa.png)
