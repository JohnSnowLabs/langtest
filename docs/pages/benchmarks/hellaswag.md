---
layout: docs
header: true
seotitle: HellaSwag Benchmark | LangTest | John Snow Labs
title: HellaSwag
key: benchmarks-hellaswag
permalink: /docs/pages/benchmarks/hellaswag/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472/)

HellaSWAG is a dataset for studying grounded commonsense inference. The samples start with one ore two sentence and the last sentence is left incomplete, there are some possible senseful completions in the dataset and model's completion is compared to them.

You can see which subsets and splits are available below.

{:.table2}
| Split                    | Details                                                                                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **test**      | Dev set Training set from the hellaswag dataset with 3000 examples which is a benchmark for Commonsense NLI. It includes a context and some endings which complete the context. |
| **test-tiny** | Truncated version of the test set from the hellaswag dataset with 50 examples.                                                                                                  |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | 	add_slangs  | - | 	A huge crowd is in the stands in an arena. A man throws a javelin. Photographers take pictures in the background. several men | - |A humongous crowd is in the stands in an arena. A bloke throws a javelin. Photographers take pictures in the background. several men | and women cheer as the javelin sails through the air. <br><br>The javelin lands with a thud, and the crowd erupts in applause. | and women cheer as the javelin sails through the air. <br><br>The javelin lands with a thud in the center of the field, and the crowd erupts in applause.  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
