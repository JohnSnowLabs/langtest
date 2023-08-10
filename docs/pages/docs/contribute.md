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

### Span Tracking

In Named Entity Recognition (NER) tasks, accurately identifying entities within text is crucial. This requires keeping track of spans, which are the positions of the original word and the transformed word within the sentence. Accurate span recognition improves NER system performance and ensures precise entity identification.

```python
    transformations.append(
        Transformation(
            original_span=Span(
                start=match.start(),
                end=match.end(),
                word=original_word,
            ),
            new_span=Span(
                start=match.start(),
                end=match.start() + len(transformed_word),
                word=transformed_word,
            ),
            ignore=False,
        )
    )
```

> Span tracking is only necessary for text classification and NER tests. For question answering and summarization tests, span tracking is not required.

### Feature Implementation Guidelines
- If your feature requires using specific data, you can add it to the **constants.py** file as a variable (e.g., a dictionary or a list).

- Write the class name and it's corresponding alias name in the **constants.py** file within the `DEFAULT_PERTURBATIONS` and `PERTURB_CLASS_MAP` sections.

- When contributing to LangTest, kindly ensure new features are added without introducing external dependencies

### Testing Your Feature

It is a good practive that you should add necessary tests after making any feature update or a fix etc.

- You can find all of the test files neatly organized within the designated `test` directory. To ensure comprehensive testing coverage, it is of utmost importance that you create a test that corresponds specifically to your feature. 

- This test should be integrated into the **test_robustness.py** file, which serves as a central hub for assessing the resilience and effectiveness of various components. 

```python
    def test_your_test_name(self) -> None:
        """
        Test the Yourclassname ...

        """
        transformed_samples = Yourclassname.transform(self.sentences)
        # add assertions to validate the transformed_samples
```