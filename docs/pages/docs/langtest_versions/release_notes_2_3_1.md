---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_2_3_1
key: docs-release-notes
modify_date: 2024-12-02
---

<div class="h3-box" markdown="1">

## 2.3.1
------------------
## Description

In this patch version, we've resolved several critical issues to enhance the functionality and bugs in the **LangTest** developed by JohnSnowLabs. Key fixes include correcting the NER task evaluation process to ensure that cases with empty expected results and non-empty predictions are appropriately flagged as failures. We've also addressed issues related to exceeding training dataset limits during test augmentation and uneven allocation of augmentation data across test cases. Enhancements include improved template generation using the OpenAI API, with added validation in the Pydantic model to ensure consistent and accurate outputs. Additionally, the integration of Azure OpenAI service for template-based augmentation has been initiated, and the issue with the Sphinx API documentation has been fixed to display the latest version correctly.

## 🐛 Fixes
- **NER Task Evaluation Fixes:**
  - Fixed an issue where NER evaluations passed incorrectly when expected results were empty, but actual results contained predictions. This should have failed. [#1076]
  - Fixed an issue where NER predictions had differing lengths between expected and actual results. [#1076]
 - **API Documentation Link Broken**: 
   - Fixed an issue where Sphinx API documentation wasn't showing the latest version docs. [#1077]
- **Training Dataset Limit Issue:**
  - Fixed the issue where the maximum limit set on the training dataset was exceeded during test augmentation allocation. [#1085]
- **Augmentation Data Allocation:**
  - Fixed the uneven allocation of augmentation data, which resulted in some test cases not undergoing any transformations. [#1085]
- **DataAugmenter Class Issues:**
  - Fixed issues where export types were not functioning as expected after data augmentation. [#1085]
- **Template Generation with OpenAI API:**
  - Resolved issues with OpenAI API when generating different templates from user-provided ones, which led to invalid outputs like paragraphs or incorrect JSON. Implemented structured outputs to resolve this. [#1085]

## ⚡ Enhancements
- **Pydantic Model Enhancements:**
  - Added validation steps in the Pydantic model to ensure templates are generated as required. [#1085]
- **Azure OpenAI Service Integration:**
  - Implemented the template-based augmentation using Azure OpenAI service. [#1090]
- **Text Classification Support:**
  - Support for multi-label classification in text classification tasks is added. [#1096]
 - **Data Augmentation**:
   - Add JSON Output for NER Sample to Support Generative AI Lab[#1099][#1100]

## What's Changed
* chore: reapply transformations to NER task after importing test cases by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1076
* updated the python api documentation with sphinx by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1077
* Patch/2.3.1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1078
* Bug/ner evaluation fix in is_pass() by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1080
* resolved: recovering the transformation object. by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1081
* fixed: consistent issues in augmentation by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1085
* Chore: Add Option to Configure Number of Generated Templates in Templatic Augmentation by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1089
* resolved/augmentation errors  by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1090
* Fix/augmentations by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1091
* Feature/add support for the multi label classification model by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1096
* Patch/2.3.1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1097
* chore: update pyproject.toml version to 2.3.1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1098
* chore: update DataAugmenter to support generating JSON output in GEN AI LAB by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1100
* Patch/2.3.1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1101
* implemented: basic version to handling document wise. by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1094
* Fix/module error with openai package by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1102
* Patch/2.3.1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1103


**Full Changelog**: https://github.com/JohnSnowLabs/langtest/compare/2.3.0...2.3.1

</div>
{%- include docs-langtest-pagination.html -%}
