---
layout: docs
header: true
seotitle: ASDiv Benchmark | LangTest | John Snow Labs
title: ASDiv
key: benchmarks-asdiv
permalink: /docs/pages/benchmarks/asdiv/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---
Source: [A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers](https://arxiv.org/abs/2106.15772)

The ASDiv benchmark is a dataset of math word problems (MWPs) designed to evaluate the capability of various MWP solvers. The dataset is diverse in terms of both language patterns and problem types. The dataset is intended to be a challenging benchmark for natural language processing models and to encourage the development of models that can perform complex reasoning and inference.

You can see which subsets and splits are available below.

{:.table2}
| Split           |Details                                                                                                |
| ----------------|------------------------------------------------------------------------------------------------------ |
| **test**       | Testing set from the ASDiv dataset, containing 1000 question answers examples.                         |
| **test-tiny**  | Truncated version of the test set from the ASDiv dataset, containing 50 question and answers examples. |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_ocr_typo | Seven red apples and two green apples are in the basket. | How many apples are in the basket? | zeien r^ed apples an^d tvO grcen apples are i^n t^e basket. | ho^w m^any apples are i^n t^he basket? | Nine apples are in the basket. | Four apples are in the basket. | False  |


> Generated Results for `text-davinci-003` model from `OpenAI`
