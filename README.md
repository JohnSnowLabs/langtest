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
!pip install langtest

# Import and create a Harness object
from langtest import Harness
h = Harness(task='ner', model='dslim/bert-base-NER', hub='huggingface')

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

