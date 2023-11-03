<div align="center">
  <h1 style="text-align: center; vertical-align: middle;">LangTest: Deliver Safe & Effective Language Models</h1>
</div>

<p align="center">
    <a href="https://github.com/JohnSnowLabs/langtest/releases">
        <img alt="Release Notes" src="https://img.shields.io/github/v/release/johnsnowlabs/langtest.svg">
    </a>
    <a href="https://www.johnsnowlabs.com/responsible-ai-blog/">
        <img alt="Blog" src="https://img.shields.io/badge/Responsible AI Blogs-8A2BE2">
    </a>
    <a href="https://langtest.org/docs/pages/docs/install">
        <img alt="Documentation" src="https://img.shields.io/website?up_message=online&url=https%3A%2F%2Flangtest.org%2F">
    </a>
    <a href="https://star-history.com/#JohnSnowLabs/langtest">
        <img alt="GitHub star chart" src="https://img.shields.io/github/stars/JohnSnowLabs/langtest?style=social">
    </a>
    <a href="https://github.com/JohnSnowLabs/langtest/issues">
        <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/JohnSnowLabs/langtest">
    </a>
    <a href="https://pepy.tech/project/langtest">
        <img alt="Downloads" src="https://static.pepy.tech/badge/langtest">
    </a>
    <a href="https://github.com/JohnSnowLabs/langtest/actions/workflows/build_and_test.yml">
        <img alt="CI" src="https://github.com/JohnSnowLabs/langtest/actions/workflows/build_and_test.yml/badge.svg">
    </a>
    <a href="https://github.com/JohnSnowLabs/langtest/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" />
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Image" />
</p>

![screenshot](https://raw.githubusercontent.com/JohnSnowLabs/langtest/gh-pages/docs/assets/images/langtest/langtest_flow_graphic.jpeg)

<p align="center">
  <a href="#project's-website">Project's Website</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#community-support">Community Support</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#mission">Mission</a> •
  <a href="#license">License</a>
</p>

## Project's Website

Take a look at our official page for user documentation and examples: [langtest.org](http://langtest.org/) 

## Key Features

- Generate and execute more than 50 distinct types of tests only with 1 line of code
- Test all aspects of model quality: robustness, bias, representation, fairness and accuracy.​
- Automatically augment training data based on test results (for select models)​
- Support for popular NLP frameworks for NER, Translation and Text-Classifcation: Spark NLP, Hugging Face & Transformers.
- Support for testing LLMS ( OpenAI, Cohere, AI21, Hugging Face Inference API and Azure-OpenAI LLMs) for question answering, toxicity, clinical-tests, legal-support, factuality, sycophancy and summarization task. 

## How To Use

```python
# Install langtest
!pip install langtest[transformers]

# Import and create a Harness object
from langtest import Harness
h = Harness(task='ner', model={"model":'dslim/bert-base-NER', "hub":'huggingface'})

# Generate test cases, run them and view a report
h.generate().run().report()
```

> **Note**
> For more extended examples of usage and documentation, head over to [langtest.org](https://www.langtest.org)

## Community Support

- [Slack](https://www.johnsnowlabs.com/slack-redirect/) For live discussion with the LangTest community, join the `#langtest` channel
- [GitHub](https://github.com/JohnSnowLabs/langtest/tree/main) For bug reports, feature requests, and contributions
- [Discussions](https://github.com/JohnSnowLabs/langtest/discussions) To engage with other community members, share ideas, and show off how you use LangTest!

## Mission

While there is a lot of talk about the need to train AI models that are safe, robust, and fair - few tools have been made available to data scientists to meet these goals. As a result, the front line of NLP models in production systems reflects a sorry state of affairs. 

We propose here an early stage open-source community project that aims to fill this gap, and would love for you to join us on this mission. We aim to build on the foundation laid by previous research such as [Ribeiro et al. (2020)](https://arxiv.org/abs/2005.04118), [Song et al. (2020)](https://arxiv.org/abs/2004.00053), [Parrish et al. (2021)](https://arxiv.org/abs/2110.08193), [van Aken et al. (2021)](https://arxiv.org/abs/2111.15512) and many others. 

[John Snow Labs](www.johnsnowlabs.com) has a full development team allocated to the project and is committed to improving the library for years, as we do with other open-source libraries. Expect frequent releases with new test types, tasks, languages, and platforms to be added regularly. We look forward to working together to make safe, reliable, and responsible NLP an everyday reality. 

## Comparing Benchmark Datasets: Use Cases and Evaluations

Langtest comes with different datasets to test your models, covering a wide range of use cases and evaluation scenarios.

| Dataset       | Use Case                                                                                           | Notebook                                                                                                                                             |
|---------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| [**BoolQ**](https://aclanthology.org/N19-1300/)    | Evaluate the ability of your model to answer boolean questions (yes/no) based on a given passage or context.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/BoolQ_dataset.ipynb)   |
| [**NQ-open**](https://aclanthology.org/Q19-1026/)   | Evaluate the ability of your model to answer open-ended questions based on a given passage or context.  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/NQ_open_dataset.ipynb)   |
| [**TruthfulQA**](https://aclanthology.org/2022.acl-long.229/) | Evaluate the model's capability to answer questions accurately and truthfully based on the provided information.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/TruthfulQA_dataset.ipynb)   |
| [**MMLU**](https://arxiv.org/abs/2009.03300)      | Evaluate language understanding models' performance in different domains. It covers 57 subjects across STEM, the humanities, the social sciences, and more.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/mmlu_dataset.ipynb)   |
| [**NarrativeQA**](https://aclanthology.org/Q18-1023/) | Evaluate your model's ability to comprehend and answer questions about long and complex narratives, such as stories or articles.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/NarrativeQA_Question_Answering.ipynb)   |
| [**HellaSwag**](https://aclanthology.org/P19-1472/) | Evaluate your model's ability in completions of sentences.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/HellaSwag_Question_Answering.ipynb)   |
| [**Quac**](https://aclanthology.org/D18-1241/)      | Evaluate your model's ability to answer questions given a conversational context, focusing on dialogue-based question-answering.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/quac_dataset.ipynb)   |
| [**OpenBookQA**](https://allenai.org/data/open-book-qa)| Evaluate your model's ability to answer questions that require complex reasoning and inference based on general knowledge, similar to an "open-book" exam.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/quac_dataset.ipynb)   |
| [**BBQ**](https://arxiv.org/abs/2110.08193)       | Evaluate how your model responds to questions in the presence of social biases against protected classes across various social dimensions. Assess biases in model outputs with both under-informative and adequately informative contexts, aiming to promote fair and unbiased question-answering models.   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/BBQ_dataset.ipynb)   |
|[**XSum**](https://aclanthology.org/D18-1206/) | Evaluate your model's ability to generate concise and informative summaries for long articles with the XSum dataset. It consists of articles and corresponding one-sentence summaries, offering a valuable benchmark for text summarization models. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/XSum_dataset.ipynb)|
|[**Real Toxicity Prompts**](https://aclanthology.org/2020.findings-emnlp.301/) | Evaluate your model's accuracy in recognizing and handling toxic language with the Real Toxicity Prompts dataset. It contains real-world prompts from online platforms, ensuring robustness in NLP models to maintain safe environments. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/OpenAI_QA_Testing_Notebook.ipynb)
|[**LogiQA**](https://aclanthology.org/2020.findings-emnlp.301/) | Evaluate your model's accuracy on Machine Reading Comprehension with Logical Reasoning questions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LogiQA_dataset.ipynb)
|[**BigBench Abstract narrative understanding**](https://arxiv.org/abs/2206.04615) | Evaluate your model's performance in selecting the most relevant proverb for a given narrative. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
|[**BigBench Causal Judgment**](https://arxiv.org/abs/2206.04615) | Evaluate your model's performance in measuring the ability to reason about cause and effect. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
|[**BigBench DisambiguationQA**](https://arxiv.org/abs/2206.04615) | Evaluate your model's performance in determining the interpretation of sentences containing ambiguous pronoun references.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](hhttps://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
|[**BigBench DisflQA**](https://arxiv.org/abs/2206.04615) | Evaluate your model's performance in picking the correct answer span from the context given the disfluent question. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
|[**ASDiv**](https://arxiv.org/abs/2106.15772) | Evaluate your model's ability to answer questions based on Math Word Problems. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/ASDiv_dataset.ipynb)
|[**Legal-QA**](https://arxiv.org/abs/2308.11462)                | Evaluate your model's performance on legal-qa datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LegalQA_Datasets.ipynb)      |
| [**CommonsenseQA**](https://arxiv.org/abs/1811.00937)                  | Evaluate your model's performance on the CommonsenseQA dataset, which demands a diverse range of commonsense knowledge to accurately predict the correct answers in a multiple-choice question answering format.  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/CommonsenseQA_dataset.ipynb)               |
| [**SIQA**](https://arxiv.org/abs/1904.09728)                 | Evaluate your model's performance by assessing its accuracy in understanding social situations, inferring the implications of actions, and comparing human-curated and machine-generated answers.  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/SIQA_dataset.ipynb)               |
| [**PIQA**](https://arxiv.org/abs/1911.11641)                          | Evaluate your model's performance on the PIQA dataset, which tests its ability to reason about everyday physical situations through multiple-choice questions, contributing to AI's understanding of real-world interactions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/PIQA_dataset.ipynb)  
| [**MultiLexSum**](https://arxiv.org/abs/2206.10883) | Evaluate your model's ability to generate concise and informative summaries for legal case contexts from the Multi-LexSum dataset, with a focus on comprehensively capturing essential themes and key details within the legal narratives. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/MultiLexSum_dataset.ipynb) |
| [**FIQA**](https://paperswithcode.com/dataset/fiqa-1) | Evaluate your model's performance on the FiQA dataset, a comprehensive and specialized resource designed for finance-related question-answering tasks. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Fiqa_dataset.ipynb) |


> **Note**
> For usage and documentation, head over to [langtest.org](https://langtest.org/docs/pages/docs/data#question-answering)


## Blog

You can check out the following langtest articles:

| Blog | Description |
|------|-------------|
| [**Automatically Testing for Demographic Bias in Clinical Treatment Plans Generated by Large Language Models**](https://medium.com/john-snow-labs/automatically-testing-for-demographic-bias-in-clinical-treatment-plans-generated-by-large-language-ffcf358b6092) | Helps in understanding and testing demographic bias in clinical treatment plans generated by LLM. |
| [**LangTest: Unveiling & Fixing Biases with End-to-End NLP Pipelines**](https://www.johnsnowlabs.com/langtest-unveiling-fixing-biases-with-end-to-end-nlp-pipelines/) | The end-to-end language pipeline in LangTest empowers NLP practitioners to tackle biases in language models with a comprehensive, data-driven, and iterative approach. |
| [**Beyond Accuracy: Robustness Testing of Named Entity Recognition Models with LangTest**](https://medium.com/john-snow-labs/beyond-accuracy-robustness-testing-of-named-entity-recognition-models-with-langtest-fb046ace7eb9) | While accuracy is undoubtedly crucial, robustness testing takes natural language processing (NLP) models evaluation to the next level by ensuring that models can perform reliably and consistently across a wide array of real-world conditions. |
| [**Elevate Your NLP Models with Automated Data Augmentation for Enhanced Performance**](https://medium.com/john-snow-labs/elevate-your-nlp-models-with-automated-data-augmentation-for-enhanced-performance-71aa7812c699) | In this article, we discuss how automated data augmentation may supercharge your NLP models and improve their performance and how we do that using  LangTest. |
| [**Mitigating Gender-Occupational Stereotypes in AI: Evaluating Models with the Wino Bias Test through Langtest Library**](https://www.johnsnowlabs.com/mitigating-gender-occupational-stereotypes-in-ai-evaluating-language-models-with-the-wino-bias-test-through-the-langtest-library/) | In this article, we discuss how we can test the "Wino Bias” using LangTest. It specifically refers to testing biases arising from gender-occupational stereotypes. |
| [**Automating Responsible AI: Integrating Hugging Face and LangTest for More Robust Models**](https://www.johnsnowlabs.com/automating-responsible-ai-integrating-hugging-face-and-langtest-for-more-robust-models/) | In this article, we have explored the integration between Hugging Face, your go-to source for state-of-the-art NLP models and datasets, and LangTest, your NLP pipeline’s secret weapon for testing and optimization. |
| [**Detecting and Evaluating Sycophancy Bias: An Analysis of LLM and AI Solutions**](https://medium.com/john-snow-labs/detecting-and-evaluating-sycophancy-bias-an-analysis-of-llm-and-ai-solutions-ce7c93acb5db) | In this blog post, we discuss the pervasive issue of sycophantic AI behavior and the challenges it presents in the world of artificial intelligence. We explore how language models sometimes prioritize agreement over authenticity, hindering meaningful and unbiased conversations. Furthermore, we unveil a potential game-changing solution to this problem, synthetic data, which promises to revolutionize the way AI companions engage in discussions, making them more reliable and accurate across various real-world conditions. |
| [**Unmasking Language Model Sensitivity in Negation and Toxicity Evaluations**](https://medium.com/john-snow-labs/unmasking-language-model-sensitivity-in-negation-and-toxicity-evaluations-f835cdc9cabf) | In this blog post, we delve into Language Model Sensitivity, examining how models handle negations and toxicity in language. Through these tests, we gain insights into the models' adaptability and responsiveness, emphasizing the continuous need for improvement in NLP models. |
| [**Unveiling Bias in Language Models: Gender, Race, Disability, and Socioeconomic Perspectives**](https://medium.com/john-snow-labs/unveiling-bias-in-language-models-gender-race-disability-and-socioeconomic-perspectives-af0206ed0feb) | In this blog post, we explore bias in Language Models, focusing on gender, race, disability, and socioeconomic factors. We assess this bias using the CrowS-Pairs dataset, designed to measure stereotypical biases. To address these biases, we discuss the importance of tools like LangTest in promoting fairness in NLP systems. |
| [**Unmasking the Biases Within AI: How Gender, Ethnicity, Religion, and Economics Shape NLP and Beyond**](https://medium.com/@chakravarthik27/cf69c203f52c) | In this blog post, we tackle AI bias on how Gender, Ethnicity, Religion, and Economics Shape NLP systems. We discussed strategies for reducing bias and promoting fairness in AI systems. |
> **Note**
> To checkout all blogs, head over to [Blogs](https://www.johnsnowlabs.com/responsible-ai-blog/)

## Contributing

We welcome all sorts of contributions:

- [Ideas](https://github.com/JohnSnowLabs/langtest/discussions/categories/ideas)
- [Discussions](https://github.com/JohnSnowLabs/langtest/discussions)
- [Feedback](https://github.com/JohnSnowLabs/langtest/discussions/categories/general)
- [Documentation](https://www.example.com/documentation)
- [Bug reports](https://www.example.com/bug-reports)

Feel free to clone the repo and submit pull-requests! You can also contribute by simply opening an [issue](https://github.com/JohnSnowLabs/langtest/issues) or [discussion](https://github.com/JohnSnowLabs/langtest/discussions) in this repo.

## Contributors

We would like to acknowledge all contributors of this open-source community project. 

<a href="https://github.com/johnsnowlabs/langtest/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=johnsnowlabs/langtest" />
</a>

## License

LangTest is released under the [Apache License 2.0](https://github.com/JohnSnowLabs/langtest/blob/main/LICENSE), which guarantees commercial use, modification, distribution, patent use, private use and sets limitations on trademark use, liability and warranty.

