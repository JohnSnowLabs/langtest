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

- **`training_data`**: (Required) Specifies the source of the original training data. It should be a dictionary containing the necessary information about the dataset.

- **`augmented_data`**: (Required) Name of the file to store the augmented data. The augmented dataset will be saved in this file.

- **`custom_proportions`**: (Required) custom_proportions is a dictionary with augmentation on test type as key and proportion as value. The proportion is the percentage of the test cases that will be augmented with the given augmentation type.

- **`export_mode`**: (Optional) Specifies how the augmented data should be exported. The possible values are:
    - `'inplace'`: Modifies the list of samples in place.
    - `'add'`: Adds new samples to the input data.
    - `'transformed'`: Exports only the transformed data, excluding different untransformed samples.

```python
custom_proportions = {
    'add_typo':0.3,
    'lowercase':0.3
}

data_kwargs = {
      "data_source" : "conll03.conll",
       }

h.augment(
    training_data = data_kwargs,
    augmented_data ="augmented_conll03.conll",
    custom_proportions=custom_proportions,
    export_mode="transformed")
```

This method applies perturbations to the input data based on the recommendations from the Harness report. This augmented dataset can then be used to retrain a model so as to make it more robust than its previous version.

</div></div>