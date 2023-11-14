---
layout: docs
header: true
seotitle: Consumer-Contracts Benchmark | LangTest | John Snow Labs
title: Consumer-Contracts
key: benchmarks-consumer-contracts
permalink: /docs/pages/benchmarks/consumer-contracts/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

Source: [Answer yes/no questions on the rights and obligations created by clauses in terms of services agreements.](https://github.com/HazyResearch/legalbench/tree/main/tasks/consumer_contracts_qa)

**Consumer Contracts** contains yes/no questions relating to consumer contracts (specifically, online terms of service) - and is relevant to the legal skill of contract interpretation.

You can see which subsets and splits are available below.

{:.table2}
| Split                  | Details                                                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------|
| **test** | Test set from the Consumer-Contracts dataset, containing 396 samples.                                                |  


#### Example


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | Your content<br>Some of our services give you the opportunity to make your content publicly available for example, you might post a product or restaurant review that you wrote, or you might upload a blog post that you created.<br>See the Permission to use your content section for more about your rights in your content, and how your content is used in our services<br>See the Removing your content section to learn why and how we might remove user-generated content from our services<br>If you think that someone is infringing your intellectual property rights, you can send us notice of the infringement and well take appropriate action. For example, we suspend or close the Google Accounts of repeat copyright infringers as described in our Copyright Help Centre. | Will Google help me if I think someone has taken and used content Ive created without my permission? | Your content<br>Some of our services give u da opportunity 2 make your content publicly available 4 example, youight post a product or restaurant review that you ute, or you miu ul a blog post that you creau.<br>See tdaPermission t2use your content section fo4ore abt your rights in your content, and how your content is used in our services<br>See thdaemoving your content section to2earn why and how we might remove user-generated content from our services<br>If you think ut someone is infringing your intellectual property rights, you can sendu notice of thedafringement and well take appropriate action. For 4mple, we suspend or close the dagle Accounts of repeat copyright infringers as described in our Copyright halp Centre. |Will Google halp me if I think someone has taken and used content Ive created w/o my permission? | True | True  | True |


> Generated Results for `text-davinci-003` model from `OpenAI`
