---
layout: docs
header: true
seotitle: Privacy-Policy Benchmark | LangTest | John Snow Labs
title: Privacy-Policy
key: benchmarks-privacy-policy
permalink: /docs/pages/benchmarks/privacy-policy/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question.](https://github.com/HazyResearch/legalbench/tree/main/tasks/privacy_policy_qa)

**Privacy-Policy** is a binary classification dataset in which the LLM is provided with a question (e.g., "do you publish my data") and a clause from a privacy policy. The LLM must determine if the clause contains an answer to the question, and classify the question-clause pair as `Relevant` or `Irrelevant`.

You can see which subsets and splits are available below.

{:.table2}
| Split                  | Details                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **test**     | Test set from the Privacy-Policy dataset, containing 10923 samples.                                             |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | The information may be disclosed to: (i) provide joint content and our services (eg, registration, coordination of membership accounts between the Viber corporate family, transactions, analytics and customer support); (ii) help detect and prevent potentially illegal acts, violations of our policies, fraud and/or data security breaches. | will my personal details be shared with third party companies? | da 411 may b disclosed 2: (i) provide joint content and our services (eg, registration, coordination of membership accounts between tdaViber corporate fly, transactions, analytics and customer support); (ii) halp detect and prevent potentially illegal acts, violations of our policies, fraud and/or data security breaches. |will my personal details b shared with 3rd party companies? | False | False  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
