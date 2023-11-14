---
layout: docs
header: true
seotitle: Contracts Benchmark | LangTest | John Snow Labs
title: Contracts
key: benchmarks-contracts
permalink: /docs/pages/benchmarks/contracts/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [Answer yes/no questions about whether contractual clauses discuss particular issues.](https://github.com/HazyResearch/legalbench/tree/main/tasks/contract_qa)

**Contracts** is a binary classification dataset where the LLM must determine if language from a contract contains a particular type of content.

You can see which subsets and splits are available below.

{:.table2}
| Split                  | Details                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **test**       | Test set from the Contracts dataset, containing 80 samples.                                                   |

#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | In the event that a user's credentials are compromised, the Company shall promptly notify the affected user and require them to reset their password. The Company shall also take reasonable steps to prevent unauthorized access to the user's account and to prevent future compromises of user credentials. | Does the clause discuss compromised user credentials? | In da event that a user's credentials r compromised, tdaCompany shall promptly notify thdaffected user and require them 2 reset their password. Thedampany shall also take reasonable steps t2prevent unauthorized access to2he user's account and to 2event future compromises of user credentials. |Does da clause discuss compromised user credentials? | True	 | True  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
