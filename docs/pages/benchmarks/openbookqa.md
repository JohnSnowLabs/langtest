---
layout: docs
header: true
seotitle: OpenBookQA Benchmark | LangTest | John Snow Labs
title: OpenBookQA
key: benchmarks-openbookqa
permalink: /docs/pages/benchmarks/openbookqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/abs/1809.02789)

The OpenBookQA dataset is a collection of multiple-choice questions that require complex reasoning and inference based on general knowledge, similar to an “open-book” exam. The questions are designed to test the ability of natural language processing models to answer questions that go beyond memorizing facts and involve understanding concepts and their relations. The dataset contains 500 questions, each with four answer choices and one correct answer. The questions cover various topics in science, such as biology, chemistry, physics, and astronomy.

You can see which subsets and splits are available below.

{:.table2}
| Split                    | Details                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the OpenBookQA dataset, containing 500 multiple-choice elementary-level science questions |
| **test-tiny** | Truncated version of the test set from the OpenBookQA dataset, containing 50 multiple-choice examples.     |

Here is a sample from the dataset:

{:.table2}
| question                                                             | answer    |
| -------------------------------------------------------------------- | --------- |
| A cactus stem is used to store  A. fruit B. liquid C. food D. spines | B. liquid |
