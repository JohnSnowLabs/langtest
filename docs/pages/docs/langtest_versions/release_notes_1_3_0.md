---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_1_3_0
key: docs-release-notes
modify_date: 2023-08-11
---

<div class="h3-box" markdown="1">

## 1.3.0

## 📢 Highlights

**LangTest 1.3.0 Release by John Snow Labs** 🚀: We've amped up our support for Clinical-Tests, made it simpler to upload models and augmented datasets to HF, and ventured into the domain of Prompt-Injection tests. Streamlined codebase, bolstered unit test coverage, added support for custom column names in harness for CSVs and polished contribution protocols with bug fixes!

</div><div class="h3-box" markdown="1">

## 🔥 New Features

- Adding support for clinical-tests.
- Adding support for prompt-injection test.
- Updated Harness format.
- Adding support for model/dataset upload to HF.
- Adding contribution guidelines.
- Improving Unittest coverage.
- Adding support for custom column names in harness for csv.

## ❓ How to Use

```
pip install "langtest[langchain,openai,transformers]"

import os

os.environ["OPENAI_API_KEY"] = <ADD OPEN-AI-KEY>
```

Create your test harness in 3 lines of code :test_tube:
```
# Import and create a Harness object
from langtest import Harness

harness = Harness(task="clinical-tests",model={"model": "gpt-3.5-turbo-instruct", "hub": "openai"},data = {"data_source": "Gastroenterology-files"})

# Generate test cases, run them and view a report
h.generate().run().report()
```

## 🐛  Bug Fixes

* Fix fairness scores

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}