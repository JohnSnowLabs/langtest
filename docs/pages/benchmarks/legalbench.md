---
layout: docs
header: true
seotitle: LegalBench Benchmark | LangTest | John Snow Labs
title: LegalBench
key: benchmarks-legalbench
permalink: /docs/pages/benchmarks/legalbench/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

### LegalBench
Source: [LegalBench: A Benchmark Dataset for Legal Term Understanding in Contracts](https://arxiv.org/abs/2308.11462)

LegalBench is a collection of datasets for evaluating natural language models on legal tasks. It consists of three datasets: Consumer Contracts, Privacy Policy, and Contracts-QA. Consumer Contracts contains yes/no questions on the rights and obligations created by clauses in terms of services agreements1. Privacy Policy contains questions and clauses from privacy policies, and requires determining if the clause contains enough information to answer the question. Contracts-QA contains true/false questions about whether contractual clauses discuss particular issues. These datasets can be used to test the ability of models to understand and reason about legal language and documents.

You can see which subsets and splits are available below.

{:.table2}
| Split Name             | Details                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Consumer Contracts** | It consists of yes/no questions on the rights and obligations created by clauses in terms of services agreements.                |  
| **Privacy-Policy**     | Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question. |
| **Contracts-QA**       | Answer True/False questions about whether contractual clauses discuss particular issues.                                         |
