---
layout: docs
header: true
seotitle: PIQA Benchmark | LangTest | John Snow Labs
title: PIQA
key: benchmarks-piqa
permalink: /docs/pages/benchmarks/piqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641)

The PIQA dataset is a collection of multiple-choice questions that test the ability of language models to reason about physical commonsense in natural language. The questions are based on everyday scenarios that involve some physical knowledge, such as cooking, gardening, or cleaning. The test dataset contains 3084 questions, each with a goal, a solution, and two alternative solutions. The correct solution is the one that is most likely to achieve the goal, while the alternatives are either ineffective or harmful. The dataset is designed to challenge the models’ understanding of real-world interactions and causal effects.

You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the PIQA dataset, containing 3084 questions. This dataset does not contain labels and accuracy & fairness tests cannot be run with it.  |
| **test-tiny** | Truncated version of PIQA dataset which contains 50 questions. This dataset does not contain labels and accuracy & fairness tests cannot be run with it. |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | 	dyslexia_word_swap | - | hands\nA. is used to put on shoe \nB. is used to put on milk jug | - |hands\nA. is used too put on shoe \nB. is used too put on milk jug | A | A  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
