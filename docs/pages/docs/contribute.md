---
layout: docs
header: true
seotitle: Contribute | LangTest | John Snow Labs
title: Contribute to LangTest
key: docs-examples
permalink: /docs/pages/docs/contribute
modify_date: "2019-05-16"
---

We're thrilled that you're interested in contributing to our project. Your contributions can make a significant impact, and we appreciate your support in making [LangTest](https://github.com/JohnSnowLabs/langtest) even better. 
Imagine you've identified an area in LangTest that could use improvement or a new feature. Here's a step-by-step guide on how to make a valid contribution:

Prior to proceeding, ensure that you have meticulously reviewed each step outlined in the [Contribution file](https://github.com/JohnSnowLabs/langtest/blob/add-onboarding-materials/CONTRIBUTING.md). This preparation will equip you with the necessary knowledge to confidently make changes within your designated branch.

> Let's suppose you're eager to add a robustness test.

## Adding a New Test

1. Navigate to the **transform** directory within the project. This directory contains all the supported test categories. Choose the category you want to work on.

2. Inside the chosen category, you can create a class for example `Yourclassname` with a `transform` method responsible for transforming sentences by adding perturbations.

```python
    class Yourclassname(BaseRobustness):
        """A class for ....... """
        alias_name = "your_class_name"
        
        @staticmethod
        def transform(sample_list: List[Union[Sample, str]], prob: Optional[float] = 1.0) > List[Sample]: # params if required
            """
            Docstrings
            """
            #Your code here
            
        def yourfunction(text):
            """
            Docstrings
            """
            .
            .
            .
```
