---
layout: docs
seotitle: Quick Start | NLP Test | John Snow Labs
title: Quick Start
permalink: /docs/pages/docs/install
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

**nlptest** is an open-source Python library designed to help developers deliver safe and effective Natural Language Processing (NLP) models.
You can install **nlptest** using pip or conda.

</div><div class="h3-box" markdown="1">

<div class="heading" id="installation"> Installation </div>

**Using PyPI**:

```sh
pip install nlptest
```

**Using Conda**: 

Download from johnsnowlabs or conda-forge channel.

```sh
# Option 1: From the johnsowlabs channel
conda install -c johnsnowlabs nlptest

# Option 2: From the conda-forge channel
conda install -c conda-forge nlptest
```

> :bulb: The conda solver is slower than the mamba solver. Install mamba in the 
> conda environment first. And then replace all `conda install` commands with 
> `mamba install` to get a much faster installation of packages.
> To install mamba: `conda install -c conda-forge mamba`  

The library supports 50+ out of the box tests for **John Snow Labs**, **Hugging Face**, and **Spacy** models. These tests fall into robustness, accuracy, bias, representation and fairness test categories for NER and Text Classification models, with support for many more tasks coming soon.

<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 10px;
    padding-bottom: 20px;
  }

  #installation {
    color: #1E77B7;
  }
  
</style>

</div></div>
