---
layout: docs
header: true
seotitle: SocialIQA Benchmark | LangTest | John Snow Labs
title: SocialIQA
key: benchmarks-socialiqa
permalink: /docs/pages/benchmarks/siqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

### SocialIQA
Source: [SocialIQA: Commonsense Reasoning about Social Interactions](https://arxiv.org/abs/1904.09728)

SocialIQA is a dataset for testing the social commonsense reasoning of language models. It consists of over 1900 multiple-choice questions about various social situations and their possible outcomes or implications. The questions are based on real-world prompts from online platforms, and the answer candidates are either human-curated or machine-generated and filtered. The dataset challenges the models to understand the emotions, intentions, and social norms of human interactions.

You can see which subsets and splits are available below.

{:.table2}
| Split Name         | Details                                                                                |
| ------------------ | -------------------------------------------------------------------------------------- |
| **SIQA-test**      | Testing set from the SIQA dataset, containing 1954 question and answer examples.       |
| **SIQA-test-tiny** | Truncated version of SIQA-test dataset which contains 50 question and answer examples. |

Here is a sample from the dataset:

{:.table2}
| passage                                                 | question                                                                  | answer            |
| ------------------------------------------------------- | ------------------------------------------------------------------------- |
| Sasha set their trash on fire to get rid of it quickly. | How would you describe Sasha? A. dirty B. Very efficient C. Inconsiderate | B. Very efficient |
