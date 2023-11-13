---
layout: docs
header: true
seotitle: BigBench Benchmark | LangTest | John Snow Labs
title: BigBench
key: benchmarks-bigbench
permalink: /docs/pages/benchmarks/bigbench/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

### BigBench

Source: [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615)

BigBench is a large-scale benchmark for measuring the performance of language models across a wide range of natural language understanding and generation tasks. It consists of many subtasks, each with its own evaluation metrics and scoring system. The subsets included in LangTest are:
- Abstract-narrative-understanding
- DisambiguationQA
- DisflQA
- Causal-judgment


You can see which subsets and splits are available below.

### Abstract-narrative-understanding

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**      | Testing set from the Bigbench/Abstract Narrative Understanding dataset, containing 1000 question answers examples.                         |
| **test-tiny** | Truncated version of the test set from the Bigbench/Abstract Narrative Understanding dataset, containing 50 question and answers examples. |


### DisambiguationQA

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**                      | Testing set from the Bigbench/DisambiguationQA dataset, containing 207 question answers examples.                                          |
| **test-tiny**                 | Truncated version of the test set from the Bigbench/DisambiguationQA dataset, containing 50 question and answers examples.                 |


### DisflQA

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**                               | Testing set from the Bigbench/DisflQA dataset, containing 1000 question answers examples.                                                  |
| **test-tiny**                               | Truncated version of the test set from the Bigbench/DisflQA dataset, containing 50 question and answers examples.                          |


### Causal-judgment

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**                       | Testing set from the Bigbench/Causal Judgment dataset, containing 190 question and answers examples.                                       |
| **test-tiny**                  | Truncated version of the test set from the Bigbench/Causal Judgment dataset, containing 50 question and answers examples.                  |
