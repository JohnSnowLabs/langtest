---
layout: docs
header: true
seotitle: NLP Docs | John Snow Labs
title: Augmentation generate()
key: docs-examples
permalink: /docs/pages/docs/generate_augmentation
modify_date: "2023-03-28"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

**harness.augment()** is a method in the library that allows users to easily augment their training data. The input to the method is the path to the training dataset, and the output is the path to save the augmented dataset. There is another optional parameter **inplace** , that is a boolean to decide whether you want to modify the input file directly. It defaults to False.

```python
harness.augment(input='training_dataset', output='augmented_dataset')
```
Essentially it applies perturbations to the input data based on the recommendations from the harness reports. Then this augmented_dataset is used to retrain the original model so as to make the model more robust and improve its performance.

</div></div>