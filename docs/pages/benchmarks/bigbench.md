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

Source: [Bigbench Dataset](https://arxiv.org/abs/2206.04615)

BigBench is a large-scale benchmark for measuring the performance of language models across a wide range of natural language understanding and generation tasks. It consists of many subtasks, each with its own evaluation metrics and scoring system. The subtasks included in LangTest are:
- Abstract Narrative Understanding
- DisambiguationQA
- DisflQA
- Causal Judgement


You can see which subsets and splits are available below.

{:.table2}
| Split Name                                              | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Bigbench-Abstract-narrative-understanding-test**      | Testing set from the Bigbench/Abstract Narrative Understanding dataset, containing 1000 question answers examples.                         |
| **Bigbench-Abstract-narrative-understanding-test-tiny** | Truncated version of the test set from the Bigbench/Abstract Narrative Understanding dataset, containing 50 question and answers examples. |
| **Bigbench-DisambiguationQA-test**                      | Testing set from the Bigbench/DisambiguationQA dataset, containing 207 question answers examples.                                          |
| **Bigbench-DisambiguationQA-test-tiny**                 | Truncated version of the test set from the Bigbench/DisambiguationQA dataset, containing 50 question and answers examples.                 |
| **Bigbench-DisflQA-test**                               | Testing set from the Bigbench/DisflQA dataset, containing 1000 question answers examples.                                                  |
| **Bigbench-DisflQA-test**                               | Truncated version of the test set from the Bigbench/DisflQA dataset, containing 50 question and answers examples.                          |
| **Bigbench-Causal-judgment-test**                       | Testing set from the Bigbench/Causal Judgment dataset, containing 190 question and answers examples.                                       |
| **Bigbench-Causal-judgment-test-tiny**                  | Truncated version of the test set from the Bigbench/Causal Judgment dataset, containing 50 question and answers examples.                  |
