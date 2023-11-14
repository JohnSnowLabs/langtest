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

Source: [LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning](https://paperswithcode.com/paper/logiqa-a-challenge-dataset-for-machine)

The LogiQA dataset is a collection of questions and answers designed to test the ability of natural language processing models to perform logical reasoning. The dataset in LangTest consists of 1000 QA instances covering multiple types of deductive reasoning, sourced from expert-written questions for testing human logical reasoning. The dataset is intended to be a challenging benchmark for machine reading comprehension models and to encourage the development of models that can perform complex logical reasoning and inference. Results show that state-of-the-art neural models perform by far worse than human ceiling 1234.

You can see which subsets and splits are available below.

{:.table2}
| Split                | Details                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the LogiQA dataset, containing 1000 question answers examples.                         |
| **test-tiny** | Truncated version of the test set from the LogiQA dataset, containing 50 question and answers examples. |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_ocr_typo | In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park. These four communities are designated as cultural area, leisure area, commercial area and administrative service area. It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area. | Based on the above statement, which of the following can be derived?<br><br>A. Civic Park is north of the administrative service area.<br>B. The leisure area is southwest of the cultural area.<br>C. The cultural district is in the northeast of the business district.<br>D. The business district is southeast of the leisure area. | i^n tle planning of a neAv district in a township, i^t was decided t^o huild a lpecial communitv in tle southeast, northwest, centered on tle citizcn park. theffe f0ur communities are designated as cultural area, leisure area, c0mmercial area a^d administrative scrvicc area. It is known tiiat tle administrative scrvicc area is southwest of tle cultural area, a^d tle cultural area is southeast of tle leisure area. |Based on tbe abovc ftatcment, whicb of tbe following c^an be derived?<br><br>A. Civic Park is noith of tbe administrative servicc area.<br>B. tbe leisure area is southwest of tbe cultural area.<br>C. tbe cultural district is i^n tbe northeast of tbe busincss district.<br>D. tbe busincss district is southeast of tbe leisure area. | B. The leisure area is southwest of the cultural area. | B. The leisure area is southwest of the cultural area.  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
