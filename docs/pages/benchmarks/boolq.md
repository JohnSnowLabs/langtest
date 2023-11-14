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


#### Example

{:.table3}
| category   | test_type        | original_context                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | original_question                                                                      | perturbed_context                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | perturbed_question                                                                  | expected_result   | actual_result   | pass   |
|:-----------|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------|:----------------|:-------|
| robustness | add_abbreviation | 20 euro note -- Until now there has been only one complete series of euro notes; however a new series, similar to the current one, is being released. The European Central Bank will, in due time, announce when banknotes from the first series lose legal tender status.                                                                                                                                                                                                                                                                                                                       | is the first series 20 euro note still legal tender                                    | 20 euro note -- Until now there has been only one complete series of euro notes; however a new series, similar 2 da current one, is being released. TdaEuropean Central Bank will, in due time, announce when banknotes from thdast series lose legal tender status.                                                                                                                                                                                                                                                                                                   | is da 1st series 20 euro note still legal tender                                    | False             | False           | True   |

> Generated Results for `text-davinci-003` model from `OpenAI`
