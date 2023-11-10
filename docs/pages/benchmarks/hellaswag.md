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

### HellaSwag
HellaSWAG is a dataset for studying grounded commonsense inference. The samples start with one ore two sentence and the last sentence is left incomplete, there are some possible senseful completions in the dataset and model's completion is compared to them.

You can see which subsets and splits are available and the other details of the dataset [here](docs/pages/docs/data#question-answering).

{:.table2}
| question                                                                               | answer                                                                           |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| He adds pasta to a large pot on the stove. He starts cooking in a pan on the stove. he | "stirs it up and adds eggs.", "lets the pasta go.", "covers the pan with a lid." |
