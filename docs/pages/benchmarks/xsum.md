---
layout: docs
header: true
seotitle: XSum Benchmark | LangTest | John Snow Labs
title: XSum
key: benchmarks-xsum
permalink: /docs/pages/benchmarks/xsum/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

### XSum
Source: [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclanthology.org/D18-1206/)

The XSum dataset is a collection of news articles and their corresponding one-sentence summaries. The dataset is designed to evaluate the performance of abstractive single-document summarization systems. The goal is to create a short, one-sentence summary answering the question "What is the article about?". The dataset is available in English and is monolingual. The dataset was created by EdinburghNLP and is available on Hugging Face.


You can see which subsets and splits are available below.

{:.table2}
| Split Name         | Details                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| **XSum**           | Training & development set from the Extreme Summarization (XSum) Dataset, containing 226,711 labeled examples |
| **XSum-test**      | Test set from the Xsum dataset, containing 1,000 labeled examples                                             |
| **XSum-test-tiny** | Truncated version of the test set from the Xsum dataset, containing 50 labeled examples                       |
| **XSum-bias**      | Manually annotated bias version of the Xsum dataset, containing 382 labeled examples                          |

