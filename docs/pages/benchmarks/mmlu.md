---
layout: docs
header: true
seotitle: MMLU Benchmark | LangTest | John Snow Labs
title: MMLU
key: benchmarks-mmlu
permalink: /docs/pages/benchmarks/mmlu/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)

The MMLU dataset is a collection of questions and answers that cover many subjects across various domains, such as STEM, humanities, social sciences, and more. The dataset is designed to measure the performance of language understanding models on a wide range of tasks, such as elementary mathematics, US history, computer science, law, and more. Each sample has a question, and 4 choices and one of them is correct. The dataset can be used to evaluate the models’ ability to reason, recall facts, and apply knowledge to different scenarios.

You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **test**      | Test set from the MMLU dataset which covers 57 tasks including elementary mathematics, US history, computer science, law, and more. We took 50 samples from each tasks in the test set.                           |
| **test-tiny** | Truncated version of test set from the MMLU dataset which covers 57 tasks including elementary mathematics, US history, computer science, law, and more. We took 10 samples from each tasks in the test-tiny set. |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_speech_to_text_typo | - | Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.<br>A. 0<br>B. 4<br>C. 2<br>D. 6 | - |Find the degree for the given field extension Kew(sqrt(2), sqrt(3), sqrt(18)) over Q.<br>A. 0<br>B. 4<br>C. 2<br>D. 6 | B. 4 | B. 4  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
