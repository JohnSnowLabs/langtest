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

### PIQA
Source: [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641)

The PIQA dataset is a collection of multiple-choice questions that test the ability of language models to reason about physical commonsense in natural language. The questions are based on everyday scenarios that involve some physical knowledge, such as cooking, gardening, or cleaning. The test dataset contains 3084 questions, each with a goal, a solution, and two alternative solutions. The correct solution is the one that is most likely to achieve the goal, while the alternatives are either ineffective or harmful. The dataset is designed to challenge the models’ understanding of real-world interactions and causal effects.

You can see which subsets and splits are available below.

{:.table2}
| Split Name         | Details                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PIQA-test**      | Testing set from the PIQA dataset, containing 3084 questions. This dataset does not contain labels and accuracy & fairness tests cannot be run with it.  |
| **PIQA-test-tiny** | Truncated version of PIQA dataset which contains 50 questions. This dataset does not contain labels and accuracy & fairness tests cannot be run with it. |

Here is a sample from the dataset:

{:.table2}
| question                                    |
| ------------------------------------------- |
| Clean electronics without leaving moisture. |
A. Use window cleaner.
B. Use rubbing alcohol." |
