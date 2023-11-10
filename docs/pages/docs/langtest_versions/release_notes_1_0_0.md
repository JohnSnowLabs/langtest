---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes/release_notes_1_0_0
key: docs-release-notes
modify_date: 2023-08-11
---

<div class="h3-box" markdown="1">

## 1.0.0

## 📢 Highlights

**LangTest 1.0.0 Release by John Snow Labs** 🚀: We are very excited to release John Snow Labs' latest library: LangTest! 🚀, formerly known as NLP Test. This is our first major step towards building responsible AI.

LangTest is an open-source library for testing LLMs, NLP models and datasets from all major NLP libraries in a few lines of code. 🧪 The library has 1 goal: delivering safe & effective models into production. 🎯 


</div><div class="h3-box" markdown="1">

## 🔥 Features

* Generate & run over 50 test types in a few lines of code 💻
* Test all aspects of model quality: robustness, bias, representation, fairness and accuracy
* Automatically augment training data based on test results (for select models)​ 💪
* Support for popular NLP frameworks for NER, Translation and Text-Classifcation: Spark NLP, Hugging Face & spaCy 🎉
* Support for testing LLMS ( OpenAI, Cohere, AI21, Hugging Face Inference API and Azure-OpenAI LLMs) for question answering, toxicity and summarization tasks. 🎉

## ❓  How to Use

```
pip install langtest
```

Create your test harness in 3 lines of code :test_tube:
```
# Import and create a Harness object
from langtest import Harness
h = Harness(task='ner', model='dslim/bert-base-NER', hub='huggingface')

# Generate test cases, run them and view a report
h.generate().run().report()
```

## 📖  Documentation

* [LangTest: Documentation](https://langtest.org/docs/pages/docs/install)
* [LangTest: Notebooks](https://langtest.org/docs/pages/tutorials/tutorials)
* [LangTest: Test Types](https://langtest.org/docs/pages/tests/test)
* [LangTest: GitHub Repo](https://github.com/JohnSnowLabs/langtest)


## ❤️  Community support

* [Slack](https://www.johnsnowlabs.com/slack-redirect/) For live discussion with the LangTest community, join the `#langtest` channel
* [GitHub](https://github.com/JohnSnowLabs/langtest/tree/main) For bug reports, feature requests, and contributions
* [Discussions](https://github.com/JohnSnowLabs/langtest/discussions) To engage with other community members, share ideas, and show off how you use NLP Test!

We would love to have you join the mission :point_right: open an issue, a PR, or give us some feedback on features you'd like to see! :raised_hands: 


## 🚀 Mission
---
While there is a lot of talk about the need to train AI models that are safe, robust, and fair - few tools have been made available to data scientists to meet these goals. As a result, the front line of NLP models in production systems reflects a sorry state of affairs.

We propose here an early stage open-source community project that aims to fill this gap, and would love for you to join us on this mission. We aim to build on the foundation laid by previous research such as [Ribeiro et al. (2020)](https://arxiv.org/abs/2005.04118), [Song et al. (2020)](https://arxiv.org/abs/2004.00053), [Parrish et al. (2021)](https://arxiv.org/abs/2110.08193), [van Aken et al. (2021)](https://arxiv.org/abs/2111.15512) and many others.

[John Snow Labs](www.johnsnowlabs.com) has a full development team allocated to the project and is committed to improving the library for years, as we do with other open-source libraries. Expect frequent releases with new test types, tasks, languages, and platforms to be added regularly. We look forward to working together to make safe, reliable, and responsible NLP an everyday reality.

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}