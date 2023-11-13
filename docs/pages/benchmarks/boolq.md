---
layout: docs
header: true
seotitle: BoolQ Benchmark | LangTest | John Snow Labs
title: BoolQ
key: benchmarks-boolq
permalink: /docs/pages/benchmarks/boolq/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/)

The BoolQ dataset is a collection of yes/no questions that are naturally occurring and generated in unprompted and unconstrained settings. The dataset contains about 16k examples, each consisting of a question, a passage, and an answer. The questions are about various topics and require reading comprehension and reasoning skills to answer. The dataset is intended to explore the surprising difficulty of natural yes/no questions and to benchmark natural language understanding systems.

You can see which subsets and splits are available below.

{:.table2}
| Split               | Details                                                                                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **combined**           | Training, development & test set from the BoolQ dataset, containing 15,942 labeled examples                                                                                         |
| **dev**       | Dev set from the BoolQ dataset, containing 3,270 labeled examples                                                                                                                   |
| **dev-tiny**  | Truncated version of the dev set from the BoolQ dataset, containing 50 labeled examples                                                                                             |
| **test**      | Test set from the BoolQ dataset, containing 3,245 labeled examples. This dataset does not contain labels and accuracy & fairness tests cannot be run with it.                       |
| **test-tiny** | Truncated version of the test set from the BoolQ dataset, containing 50 labeled examples. This dataset does not contain labels and accuracy & fairness tests cannot be run with it. |
| **bias**      | Manually annotated bias version of BoolQ dataset, containing 136 labeled examples                                                                                                   |

Here is a sample from the dataset:

{:.table2}
| question                                  | passage                                                                                   | answer |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- | ------ |
| will there be a season 2 of penny on mars | Penny on M.A.R.S. -- On April 10, 2018, the production of the second season was announced | true   |
