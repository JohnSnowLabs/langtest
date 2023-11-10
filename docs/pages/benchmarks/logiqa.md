---
layout: docs
header: true
seotitle: LogiQA Benchmark | LangTest | John Snow Labs
title: LogiQA
key: benchmarks-logiqa
permalink: /docs/pages/benchmarks/logiqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

### LogiQA
Source: [LogiQA: A Dataset of Logical Reasoning Questions](https://aclanthology.org/2020.findings-emnlp.301/)

The LogiQA dataset is a collection of questions and answers designed to test the ability of natural language processing models to perform logical reasoning. The dataset in LangTest consists of 1000 QA instances covering multiple types of deductive reasoning, sourced from expert-written questions for testing human logical reasoning. The dataset is intended to be a challenging benchmark for machine reading comprehension models and to encourage the development of models that can perform complex logical reasoning and inference. Results show that state-of-the-art neural models perform by far worse than human ceiling 1234.

You can see which subsets and splits are available below.

{:.table2}
| Split Name           | Details                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| **LogiQA-test**      | Testing set from the LogiQA dataset, containing 1000 question answers examples.                         |
| **LogiQA-test-tiny** | Truncated version of the test set from the LogiQA dataset, containing 50 question and answers examples. |

Here is a sample from the dataset:

{:.table2}
| passage                                                                                                                                                                                                                                                                                                  | question                                                                                                                                           | answer      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| There are five teams participating in the game. The audience had the following comments on the results? (1) The champion is either the Shannan team or the Jiangbei team. (2) The champion is neither Shanbei nor Jiangnan. (3) The champion is Jiangnan Team. (4) The champion is not the Shannan team. | The result of the match showed that only one argument was correct, so who won the championship?  A. Shannan. B. Jiangnan. C. Shanbei. D. Jiangbei. | C. Shanbei. |
