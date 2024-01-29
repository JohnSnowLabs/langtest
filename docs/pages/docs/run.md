---
layout: docs
seotitle: Running Test Cases | LangTest | John Snow Labs
title: Running Test Cases
permalink: /docs/pages/docs/run
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The `run()` method is called after the `generate()` method and is used to run all the specified tests. It returns a pass/fail flag for each test.
There are two ways to run the tests

- Running All Test Cases at Once

    ```python 
    h.run()
    ```

- Checkpointing Mechanism for Large Test Sets

    ```python 
    h.run(checkpoint=True, batch_size=500, save_checkpoints_dir="checkpoints")
    ```

    To handle large test sets effectively, you can enable checkpointing. Here's what each parameter signifies:

    - `checkpoint`: Enable this option to activate the checkpointing mechanism.
    - `batch_size`: This parameter specifies the number of test cases processed per batch.
    - `save_checkpoints_dir`: Use this option to define the directory where checkpoints and intermediate results will be saved.

    If the kernel restarts or if an API failure occurs, users can resume the execution from the last saved checkpoint, preventing the loss of already processed model responses.

    ```python 
    h = Harness.load_checkpoints(
        save_checkpoints_dir="checkpoints",
        task="text-classification",
        model={"model": "lvwerra/distilbert-imdb", "hub": "huggingface"},
    )
    h.run(checkpoint=True, batch_size=500, save_checkpoints_dir="checkpoints")         
    ```

Once the tests have been run using the `run()` method, the results can be accessed using the `.generated_results()` method. 

```python 
h.generated_results()
```

This method returns the generated results in the form of a Pandas dataframe, which provides a convenient and easy-to-use format for working with the test results. You can use this method to quickly identify the test cases that failed and to determine where fixes are needed.

A generated results dataframe looks like this:

{:.table2}
| category  | test_type |  original | test_case | expected_result |  actual_result | pass |
| - | - | - | - | - | - | - |
|robustness| lowercase | I live in Berlin | i live in berlin | Berlin: LOC | | False |
|robustness| uppercase | I live in Berlin | I LIVE IN BERLIN | Berlin: LOC | BERLIN: LOC | True |

</div>