---
layout: docs
header: true
seotitle: NQ-open Benchmark | LangTest | John Snow Labs
title: NQ-open
key: benchmarks-NQ-open
permalink: /docs/pages/benchmarks/nq-open/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---


Source: [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/)

The Natural Questions dataset is a large-scale collection of real user questions and answers from Wikipedia. It is designed to evaluate the performance of automatic question answering systems on open-domain questions. The dataset included in LangTest about 3,500 test examples. Each example consists of a query and one or more answers from the wikipedia page.

You can see which subsets and splits are available below.

{:.table2}
| Split                 | Details                                                                                            |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| **combined**           | Training & development set from the NaturalQuestions dataset, containing 3,569 labeled examples    |
| **test**      | Development set from the NaturalQuestions dataset, containing 1,769 labeled examples               |
| **test-tiny** | Training, development & test set from the NaturalQuestions dataset, containing 50 labeled examples |

Here is a sample from the dataset:

{:.table2}
| question                                | answer                |
| --------------------------------------- | --------------------- |
| when was me talk pretty one day written | "May 2, 2000", "2000" |
