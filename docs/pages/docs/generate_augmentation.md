---
layout: docs
header: true
seotitle: Data Augmentations | NLP Test | John Snow Labs
title: Data Augmentations
key: docs-install
permalink: /docs/pages/docs/generate_augmentation
modify_date: "2023-03-28"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

<div class="heading" id="generate-augmentation">Generating Augmentations</div>


The library provides a method called **`.augment()`** that facilitates the data augmentation process for training data. To use this method, you need to specify two parameters: **`input_path`**, which is the path to the original training dataset, and **`output_path`**, which is the path to save the augmented dataset. Additionally, there's an optional parameter **`inplace`**, which is a boolean that controls whether the original input file should be modified directly. By default, inplace is set to False. If True, the list of samples are modified in place. Otherwise, new samples are added to the input data. 



```python

# generating augmentations
h.augment(input_path='training_dataset', output_path='augmented_dataset')

```

Essentially it applies perturbations to the input data based on the recommendations from the harness reports. Then this augmented_dataset is used to retrain the original model so as to make the model more robust and improve its performance.

<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 20px;
    padding-bottom: 30px;
  }

  #generate-augmentation {
    color: #1E77B7;
  }
  
</style>


</div></div>