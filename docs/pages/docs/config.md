---
layout: docs
seotitle: Config | NLP Test | John Snow Labs
title: Config
permalink: /docs/pages/docs/config
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Configuring Tests

The configuration for the tests can be passed in the form of a YAML file to the `config` parameter in the `Harness`, or by using the `configure()` method.

#### Using the YAML Configuration File

```bash
tests:
  defaults:
    min_pass_rate: 0.65
    min_score: 0.8
  robustness:
    lowercase:
      min_pass_rate: 0.60
    uppercase:
      min_pass_rate: 0.60
   bias:
     replace_to_female_pronouns
   accuracy:
    min_f1_score:
      min_score: 0.8
```

```python
from nlptest import Harness

# Create test Harness with config file
h = Harness(task='text-classification', model='path/to/local_saved_model', hub='spacy', data='test.csv', config='config.yml')
```

#### Using the `.configure()` Method

```python
from nlptest import Harness

# Create test Harness without config file
h = Harness(task='text-classification', model='path/to/local_saved_model', hub='spacy', data='test.csv')

h.configure(
  {
    'tests': {
      'defaults': {
          'min_pass_rate': 0.65
          'min_score': 0.8
      },
      'robustness': {
          'lowercase': { 'min_pass_rate': 0.60 }, 
          'uppercase': { 'min_pass_rate': 0.60 }
        },
      'bias': {
          'replace_to_female_pronouns'
        },
      'accuracy': {
          'min_f1_score'
      }
      }
  }
 )
```

</div><div class="h3-box" markdown="1">

### Config for LLMs

The configuration for LLM-related tests requires extra fields. These are .

#### Using the YAML Configuration File

```bash
tests:
  defaults:
    min_pass_rate: 0.65
    min_score: 0.8
  robustness:
    lowercase:
      min_pass_rate: 0.60
    uppercase:
      min_pass_rate: 0.60
   bias:
     replace_to_female_pronouns
   accuracy:
    min_f1_score:
      min_score: 0.8
```

</div><div class="h3-box" markdown="1">

</div></div>