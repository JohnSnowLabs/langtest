---
layout: docs
header: true
seotitle: MedicationQA Benchmark | LangTest | John Snow Labs
title: MedicationQA
key: benchmarks-medicationqa
permalink: /docs/pages/benchmarks/medical/medicationqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

**Source:** [Bridging the Gap Between Consumers' Medication Questions and Trusted Answers](https://pubmed.ncbi.nlm.nih.gov/31437878/)

The MedicationQA dataset consists of commonly asked consumer questions about medications. It includes annotations corresponding to drug focus and interactions. LangTest now integrates MedicationQA for thorough evaluation of models in medication-related scenarios.

{:.table2}
| subsets       | Details                                                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MedicationQA-Test**    | This dataset does not contain labels, so accuracy and fairness tests cannot be run on it. Only robustness tests can be applied.                             |

</div>