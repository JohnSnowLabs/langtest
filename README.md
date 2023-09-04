# LangTest: Deliver Safe & Effective Language Models

<p align="center">
    <a href="https://github.com/JohnSnowLabs/langtest/actions" alt="build">
        <img src="https://github.com/JohnSnowLabs/langtest/workflows/build/badge.svg" /></a>
    <a href="https://github.com/JohnSnowLabs/langtest/releases" alt="Current Release Version">
        <img src="https://img.shields.io/github/v/release/JohnSnowLabs/langtest.svg?style=flat-square&logo=github" /></a>
    <a href="https://github.com/JohnSnowLabs/langtest/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
    <a href="https://pypi.org/project/langtest/" alt="PyPi downloads">
        <img src="https://static.pepy.tech/personalized-badge/langtest?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads" /></a>
</p>


<p align="center">
  <a href="#project's-website">Project's Website</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#community-support">Community Support</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#mission">Mission</a> •
  <a href="#license">License</a>
</p>

![screenshot](https://raw.githubusercontent.com/JohnSnowLabs/langtest/gh-pages/docs/assets/images/langtest/langtest_flow_graphic.jpeg)

## Project's Website

Take a look at our official page for user documentation and examples: [langtest.org](http://langtest.org/) 

## Key Features

- Generate and execute more than 50 distinct types of tests only with 1 line of code
- Test all aspects of model quality: robustness, bias, representation, fairness and accuracy.​
- Automatically augment training data based on test results (for select models)​
- Support for popular NLP frameworks for NER, Translation and Text-Classifcation: Spark NLP, Hugging Face & Transformers.
- Support for testing LLMS ( OpenAI, Cohere, AI21, Hugging Face Inference API and Azure-OpenAI LLMs) for question answering, toxicity and summarization task. 

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
|[**BigBench DisambiguationQA**](https://arxiv.org/abs/2206.04615) | Evaluate your model's performance on determining the interpretation of sentences containing ambiguous pronoun references.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](hhttps://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
|[**BigBench DisflQA**](https://arxiv.org/abs/2206.04615) | Evaluate your model's performance in picking the correct answer span from the context given the disfluent question. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
|[**ASDiv**](https://arxiv.org/abs/2106.15772) | Evaluate your model's ability answer questions based on Math Word Problems. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/ASDiv_dataset.ipynb)

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

> **Note**
> To checkout all blogs, head over to [Blogs](https://www.johnsnowlabs.com/responsible-ai-blog/)

## Contributing

We welcome all sorts of contributions:

- Ideas
- Feedback
- Documentation
- Bug reports
- Development and testing

Feel free to clone the repo and submit pull-requests! You can also contribute by simply opening an issue or discussion in this repo.

## Contributors

We would like to acknowledge all contributors of this open-source community project. 

<a href="https://github.com/johnsnowlabs/langtest/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=johnsnowlabs/langtest" />
</a>

## License

LangTest is released under the [Apache License 2.0](https://github.com/JohnSnowLabs/langtest/blob/main/LICENSE), which guarantees commercial use, modification, distribution, patent use, private use and sets limitations on trademark use, liability and warranty.

