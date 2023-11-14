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

Source: [SocialIQA: Commonsense Reasoning about Social Interactions](https://arxiv.org/abs/1904.09728)

SocialIQA is a dataset for testing the social commonsense reasoning of language models. It consists of over 1900 multiple-choice questions about various social situations and their possible outcomes or implications. The questions are based on real-world prompts from online platforms, and the answer candidates are either human-curated or machine-generated and filtered. The dataset challenges the models to understand the emotions, intentions, and social norms of human interactions.

You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                |
| ------------------ | -------------------------------------------------------------------------------------- |
| **test**      | Testing set from the SIQA dataset, containing 1954 question and answer examples.       |
| **test-tiny** | Truncated version of SIQA-test dataset which contains 50 question and answer examples. |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | dyslexia_word_swap | Tracy didn't go home that evening and resisted Riley's attacks. | What does Tracy need to do before this?\nA. make a new plan\nB. Go home and see Riley\nC. Find somewhere to go | Tracy didn't go home that evening and resisted Riley's attacks. |What does Tracy need too do before this?\nA. make a new plan\nB. Go home and sea Riley\nC. Find somewhere too go | C. Find somewhere to go | A. Make a new plan	  | False |


> Generated Results for `text-davinci-003` model from `OpenAI`
