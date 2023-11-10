---
layout: docs
header: true
seotitle: NarrativeQA Benchmark | LangTest | John Snow Labs
title: NarrativeQA
key: benchmarks-narrativeqa
permalink: /docs/pages/benchmarks/narrativeqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

### NarrativeQA
Source: [The NarrativeQA Reading Comprehension Challenge](https://aclanthology.org/Q18-1023/)

The NarrativeQA dataset is a collection of stories and questions designed to test reading comprehension, especially on long documents. The dataset contains many stories from various genres, such as books, movie scripts, and news articles. For each story, there are multiple questions and answers that require understanding the plot, characters, and events of the story. The dataset is challenging because the questions are not answerable by simple keyword matching or extraction, but require inference and reasoning based on the whole story.

You can see which subsets and splits are available below.

{:.table2}
| Split Name                | Details                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NarrativeQA-test**      | Testing set from the NarrativeQA dataset, containing 3000 stories and corresponding questions designed to test reading comprehension, especially on long documents. |
| **NarrativeQA-test-tiny** | Truncated version of NarrativeQA dataset which contains 50 stories and corresponding questions examples.                                                            |

Here is a sample from the dataset:

{:.table2}
| question                        | answer                      | passage                                                                                                                               |
| ------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| In what city is Laura's prison? | "Pittsburgh.", "Pittsburgh" | Lara Brennan (Elizabeth Banks) is convicted of murdering her boss and is sentenced to life in prison. The evidence seems impossibl... |
