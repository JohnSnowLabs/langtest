# NLP Test

**Deliver Safe & Effective NLP models**

<p align="center">
    <a href="https://github.com/JohnSnowLabs/nlptest/releases" alt="Current Release Version">
        <img src="https://img.shields.io/github/v/release/JohnSnowLabs/nlptest.svg?style=flat-square&logo=github" /></a>
    <a href="https://github.com/JohnSnowLabs/nlptest/blob/master/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
    <a href="https://pypi.org/project/nlptest/" alt="PyPi downloads">
        <img src="https://static.pepy.tech/personalized-badge/nlptest?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads" /></a>
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

![screenshot](https://raw.githubusercontent.com/JohnSnowLabs/nlptest/main/docs/assets/images/nlptest/nlptest_flow_graphic.png?token=GHSAT0AAAAAABPNZRXYN3KIUTRXHIDR3GWIZAXL7OQ)

## Project's Website

Take a look at our official page for user documentation and examples: [nlptest.org](http://nlptest.org/) 

## Key Features

- Generate & run over 50 test types with 1 line of code
- Test all aspects of model quality: robustness, bias, representation, fairness and accuracy
- Automatically augment training data based on test results
- Support for popular NLP libraries: Spark NLP, Hugging Face Transformers & spaCy 
- Support for popular NLP tasks: Named Entity Recognition and Text Classification

## How To Use

```python
# Install nlptest
!pip install nlptest

# Import and create a Harness object
from nlptest import Harness
h = Harness(task='ner', model='dslim/bert-base-NER', hub='transformers')

# Generate test cases, run them and view a report
h.generate().run().report()
```

> **Note**
> For more extended examples of usage and documentation, head over to [nlptest.org](https://www.nlptest.org)

## Community Support

- [Slack](https://www.johnsnowlabs.com/slack-redirect/) For live discussion with the NLP Test community, join the `#nlptest` channel
- [GitHub](https://github.com/JohnSnowLabs/nlptest/tree/main) For bug reports, feature requests, and contributions
- [Discussions](https://github.com/JohnSnowLabs/nlptest/discussions) To engage with other community members, share ideas, and show off how you use NLP Test!

## Contributing

We welcome all sorts of contributions:

- ideas
- feedback
- documentation
- bug reports
- development and testing

Feel free to clone the repo and submit your pull-requests! Or directly create issues in this repo.

## Mission

While there is a lot of talk about the need to train AI models that are safe, robust, and equitable - few tools have been made available to data scientists to meet these goals. As a result, the front line of NLP models in production systems reflects a sorry state of affairs. We propose here an early stage open-source community project which you are welcome to join. 

[John Snow Labs](www.johnsnowlabs.com) has a full development team allocated to the project and is committed to improving the library for years, as we do with other open-source libraries. Expect frequent releases with new test types, tasks, languages, and platforms to be added regularly. We look forward to working together to make safe, reliable, and responsible NLP an everyday reality. 

## License

NLP Test is released under the [Apache License 2.0](https://github.com/JohnSnowLabs/nlptest/blob/main/LICENSE), which guarantees commercial use, modification, distribution, patent use, private use and sets limitations on trademark use, liability and warranty.

