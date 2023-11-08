---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes/release_notes_1_1_0
key: docs-release-notes
modify_date: 2023-08-11
---

<div class="h3-box" markdown="1">

## 1.1.0

## 📢 Highlights

**LangTest 1.1.0 Release by John Snow Labs** 🚀: LangTest 1.1.0 🚀 comes with brand new features, including: new capabilities to run different types of toxicity tests (lgbtqphobia, ideology, racism, xenophobia, sexism), support for doing templatic augmentations, extending support for HF datasets for summarization, support for BBQ-data, custom-replacement dicts for representation tests,  CSV augmentations for text classification, using poetry as a dependency manager and adding new robustness tests (adjective-swapping and strip-all-punctuation) with many other enhancements and bug fixes!

</div><div class="h3-box" markdown="1">

## 🔥 New Features

* Adding support for improved toxicity tests.
* Adding support for templatic augmentations.
* Adding support for strip_all_punctuation test.
* Adding support for adjective-swap tests.
* Adding support for custom replacement dictionaries for representation and bias tests.
* Adding support for BBQ-data.
* Adding support for CSV augmentations in text classification task.
* Adding support for hf datasets for summarization.
* Adding poetry as a dependency manager.
* Adding support for listing all available tests.
* Adding support for enabling user to only install the backend libraries needed.

## 🐛  Bug Fixes

* Model hub handler.
* Fixing augmentations for swap-entities.
* add_contraction bug for QA/Sum.

----------------
## 📖  Documentation

* [LangTest: Documentation](https://langtest.org/docs/pages/docs/install)
* [LangTest: Notebooks](https://langtest.org/docs/pages/tutorials/tutorials)
* [LangTest: Test Types](https://langtest.org/docs/pages/tests/test)
* [LangTest: GitHub Repo](https://github.com/JohnSnowLabs/langtest)

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}