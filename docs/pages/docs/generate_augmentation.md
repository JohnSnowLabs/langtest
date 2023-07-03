---
layout: docs
header: true
seotitle: Generating Augmentations | LangTest | John Snow Labs
title: Generating Augmentations
key: docs-install
permalink: /docs/pages/docs/generate_augmentation
modify_date: "2023-03-28"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The library provides a `augment()` method that facilitates the data augmentation process. 

Several parameters are available: 
- **`input_path`**, which is the path to the original training dataset to be augmented
- **`output_path`**, which is the path to save the augmented dataset
- **`inplace`** which is an optional parameter that controls whether the original input file should be augmented by duplicating rows in the dataset. By default, inplace is set to False. If True, the rows are modified in place and the length of the dataset remains similar. Otherwise, new rows are added to the dataset.

```python
# Generating augmentations
h.augment(input_path='training_dataset', output_path='augmented_dataset', inplace=False)
```

This method applies perturbations to the input data based on the recommendations from the Harness report. This augmented dataset can then be used to retrain a model so as to make it more robust than its previous version.

</div></div>