---
layout: docs
seotitle: NLP Docs | John Snow Labs
title: Running the testcases
permalink: /docs/pages/docs/run
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

Called after **.generate()** method and is to used to run all the specified tests. Returns a pass/fail flag for each test.

```python 
harness.run()
```

To get the run results in the form of a pandas dataframe we can use the **.generated_results()** method.

```python 
harness.generated_results()
```

 It returns the generated results in the form of a dataframe with pass/fail flag for each test that we had specified.

</div></div>