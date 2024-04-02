---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_1_8_0
key: docs-release-notes
modify_date: 2023-10-17
---

<div class="h3-box" markdown="1">

## 1.8.0

## 📢 Highlights

**LangTest 1.8.0 Release by John Snow Labs 🚀**

We're thrilled to unveil the latest advancements in LangTest with version 1.8.0. This release is centered around optimizing the codebase with extensive refactoring, enriching the debugging experience through the implementation of error codes, and enhancing workflow efficiency with streamlined task organization. The new categorization approach significantly improves the user experience, ensuring a more cohesive and organized testing process. This update also includes advancements in open source community standards, insightful blog posts, and multiple bug fixes, further solidifying LangTest's reputation as a versatile and user-friendly language testing and evaluation library.

##  🔥 Key Enhancements:

- **Optimized Codebase**: This update features a comprehensively refined codebase, achieved through extensive refactoring, resulting in enhanced efficiency and reliability in our testing processes.

- **Advanced Debugging Tools**: The introduction of error codes marks a significant enhancement in the debugging experience, addressing the previous absence of standardized exceptions. This inconsistency in error handling often led to challenges in issue identification and resolution. The integration of a unified set of standardized exceptions, tailored to specific error types and contexts, guarantees a more efficient and seamless troubleshooting process.

- **Task Categorization**: This version introduces an improved task organization system, offering a more efficient and intuitive workflow. Previously, it featured a wide range of tests such as sensitivity, clinical tests, wino-bias and many more, each treated as separate tasks. This approach, while comprehensive, could result in a fragmented workflow. The new categorization method consolidates these tests into universally recognized NLP tasks, including Named Entity Recognition (NER), Text Classification, Question Answering, Summarization, Fill-Mask, Translation, and Test Generation. This integration of tests as sub-categories within these broader NLP tasks enhances clarity and reduces potential overlap.

- **Open Source Community Standards**: With this release, we've strengthened community interactions by introducing issue templates, a code of conduct, and clear repository citation guidelines. The addition of GitHub badges enhances visibility and fosters a collaborative and organized community environment.

- **Parameter Standardization**: Aiming to bring uniformity in dataset organization and naming, this feature addresses the variation in dataset structures within the repository. By standardizing key parameters like 'datasource', 'split', and 'subset', we ensure a consistent naming convention and organization across all datasets, enhancing clarity and efficiency in dataset usage.

## 🚀 Community Contributions:
Our team has published three enlightening blogs on Hugging Face's community platform, focusing on bias detection, model sensitivity, and data augmentation in NLP models:

1. [Detecting and Evaluating Sycophancy Bias: An Analysis of LLM and AI Solutions](https://huggingface.co/blog/Rakshit122/sycophantic-ai)
2. [Unmasking Language Model Sensitivity in Negation and Toxicity Evaluations](https://huggingface.co/blog/Prikshit7766/llms-sensitivity-testing)
3. [Elevate Your NLP Models with Automated Data Augmentation for Enhanced Performance](https://huggingface.co/blog/chakravarthik27/boost-nlp-models-with-automated-data-augmentation)


## 🚀 New LangTest blogs :

| New Blog Posts | Description |
|----------------|-------------|
| [**Evaluating Large Language Models on Gender-Occupational Stereotypes Using the Wino Bias Test**](https://medium.com/john-snow-labs/evaluating-large-language-models-on-gender-occupational-stereotypes-using-the-wino-bias-test-2a96619b4960) | Delve into the evaluation of language models with LangTest on the WinoBias dataset, addressing AI biases in gender and occupational roles. |
| [**Streamlining ML Workflows: Integrating MLFlow Tracking with LangTest for Enhanced Model Evaluations**](https://medium.com/john-snow-labs/streamlining-ml-workflows-integrating-mlflow-tracking-with-langtest-for-enhanced-model-evaluations-4ce9863a0ff1) | Discover the revolutionary approach to ML development through the integration of MLFlow and LangTest, enhancing transparency and systematic tracking of models. |
| [**Testing the Question Answering Capabilities of Large Language Models**](https://medium.com/john-snow-labs/testing-the-question-answering-capabilities-of-large-language-models-1bc424d61740) | Explore the complexities of evaluating Question Answering (QA) tasks using LangTest's diverse evaluation methods. |
| [**Evaluating Stereotype Bias with LangTest**](https://medium.com/john-snow-labs/evaluating-stereotype-bias-with-langtest-8286af8f0f22) | In this blog post, we are focusing on using the StereoSet dataset to assess bias related to gender, profession, and race.|

----------------
## 🐛 Bug Fixes
- Fixed templatic augmentations [PR #851](https://github.com/JohnSnowLabs/langtest/pull/851)
- Resolved a bug in default configurations [PR #880](https://github.com/JohnSnowLabs/langtest/pull/880)
- Addressed compatibility issues between OpenAI (version 1.1.1) and Langchain [PR #877](https://github.com/JohnSnowLabs/langtest/pull/877)
- Fixed errors in sycophancy-test, factuality-test, and augmentation [PR #869](https://github.com/JohnSnowLabs/langtest/pull/869)

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}
