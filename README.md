# NLP Test: Deliver Safe & Effective Language Models

<p align="center">
    <a href="https://github.com/JohnSnowLabs/nlptest/actions" alt="build">
        <img src="https://github.com/JohnSnowLabs/nlptest/workflows/build/badge.svg" /></a>
    <a href="https://github.com/JohnSnowLabs/nlptest/releases" alt="Current Release Version">
        <img src="https://img.shields.io/github/v/release/JohnSnowLabs/nlptest.svg?style=flat-square&logo=github" /></a>
    <a href="https://github.com/JohnSnowLabs/nlptest/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
    <a href="https://pypi.org/project/nlptest/" alt="PyPi downloads">
        <img src="https://static.pepy.tech/personalized-badge/nlptest?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads" /></a>
    <a href="https://anaconda.org/conda-forge/nlptest" alt="Conda Version">
        <img src="https://img.shields.io/conda/vn/conda-forge/nlptest.svg?style=flat-square&color=blue&logo=conda-forge" /></a>
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

![screenshot](https://raw.githubusercontent.com/JohnSnowLabs/nlptest/main/docs/assets/images/nlptest/nlptest_flow_graphic.jpeg)

## Project's Website

Take a look at our official page for user documentation and examples: [nlptest.org](http://nlptest.org/) 

## Key Features

- Generate & run over 50 test types with 1 line of code
- Test all aspects of model quality: robustness, bias, representation, fairness and accuracy
- Automatically augment training data based on test results (for select models)
- Support for popular NLP frameworks: Spark NLP, Hugging Face Transformers, OpenAI, spaCy, Cohere, and many more 
- Support for popular NLP tasks: Question Answering, Text Classification & Named Entity Recognition

## How To Use

```python
# Install nlptest
!pip install nlptest

# Import and create a Harness object
from nlptest import Harness
h = Harness(task='ner', model='dslim/bert-base-NER', hub='huggingface')

# Generate test cases, run them and view a report
h.generate().run().report()
```

> **Note**
> For more extended examples of usage and documentation, head over to [nlptest.org](https://www.nlptest.org)

## Community Support

- [Slack](https://www.johnsnowlabs.com/slack-redirect/) For live discussion with the NLP Test community, join the `#nlptest` channel
- [GitHub](https://github.com/JohnSnowLabs/nlptest/tree/main) For bug reports, feature requests, and contributions
- [Discussions](https://github.com/JohnSnowLabs/nlptest/discussions) To engage with other community members, share ideas, and show off how you use NLP Test!

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

<a href="https://github.com/johnsnowlabs/nlptest/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=johnsnowlabs/nlptest" />
</a>

## License

NLP Test is released under the [Apache License 2.0](https://github.com/JohnSnowLabs/nlptest/blob/main/LICENSE), which guarantees commercial use, modification, distribution, patent use, private use and sets limitations on trademark use, liability and warranty.

