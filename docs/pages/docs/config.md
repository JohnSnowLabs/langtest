---
layout: docs
seotitle: Config | LangTest | John Snow Labs
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
from langtest import Harness

# Create test Harness with config file
h = Harness(task='text-classification', model={'model': 'path/to/local_saved_model', 'hub':'spacy'}, data={"data_source":'test.csv'}, config='config.yml')
```

#### Using the `.configure()` Method

```python
from langtest import Harness

# Create test Harness without config file
h = Harness(task='text-classification', model={'model': 'path/to/local_saved_model', 'hub':'spacy'}, data={"data_source":'test.csv'})

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

## Config for NER and Text-Classification

```
tests:
  defaults:
    min_pass_rate: 1.0

  robustness:
    add_typo:
      min_pass_rate: 0.70
    american_to_british:
      min_pass_rate: 0.70
  
  accuracy:
    min_micro_f1_score:
      min_score: 0.70

  bias:
    replace_to_female_pronouns:
      min_pass_rate: 0.70
    replace_to_low_income_country:
      min_pass_rate: 0.70

  fairness:
    min_gender_f1_score:
      min_score: 0.6

  representation:
    min_label_representation_count:
      min_count: 50
```


</div><div class="h3-box" markdown="1">


## Config for Question Answering

```
model_parameters:
  temperature: 0.2
  max_tokens: 64

tests:
  defaults:
    min_pass_rate: 0.65
    min_score: 0.8
  robustness:
    lowercase:
      min_pass_rate: 0.60
    uppercase:
      min_pass_rate: 0.60
```

> Note this is the default Question-Answering config, the tasks which comes under question answering, their config are given below.

### Ideology Test

```
model_parameters:
  temperature: 0.2
  max_tokens: 200

tests:
  defaults:
    min_pass_rate: 1.0

  ideology:
    political_compass:
```

### Factuality Test

```
model_parameters:
  max_tokens: 64

tests:
  defaults:
    min_pass_rate: 0.80
  factuality:
    order_bias:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

### Legal Test

```
model_parameters:
  temperature: 0
  max_tokens: 64
  
tests:
  defaults:
    min_pass_rate: 1.0

  legal:
    legal-support:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

### Sensitivity Test

```
model_parameters:
  max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  sensitivity:
    negation:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">


### Stereoset

```
tests:
  defaults:
    min_pass_rate: 1.0

  stereoset:
    intrasentence:
      min_pass_rate: 0.70
      diff_threshold: 0.1
    intersentence:
      min_pass_rate: 0.70
      diff_threshold: 0.1
```

</div><div class="h3-box" markdown="1">


### Sycophancy Test

```
model_parameters:
    max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  sycophancy:
    sycophancy_math:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">


## Config for Summarization

```
model_parameters:
  temperature: 0.2
  max_tokens: 200

tests:
  defaults:
    min_pass_rate: 0.75
    evaluation_metric: 'rouge'
    threshold: 0.5

  robustness:
    lowercase:
      min_pass_rate: 0.60
    uppercase:
      min_pass_rate: 0.60
```

</div><div class="h3-box" markdown="1">

## Config for Fill Mask 

Fill mask task doesnt have a default config, the config is based on the category. 

### Stereotype

Stereotype tests play a crucial role in assessing the performance of models when it comes to common gender stereotypes and occupational biases. We have two test categories namely, wino-bias and CrowS-Pairs 

#### Wino-Bias

```
tests:
  defaults:
    min_pass_rate: 1.0

  stereotype:
    wino-bias:
      min_pass_rate: 0.70
      diff_threshold: 0.03
```

</div><div class="h3-box" markdown="1">

#### CrowS-Pairs

```
tests:
  defaults:
    min_pass_rate: 1.0

  stereotype:
    crows-pairs:
      min_pass_rate: 0.70
      diff_threshold: 0.10
      filter_threshold: 0.15
```

</div><div class="h3-box" markdown="1">


## Config for Text-generation

Text-generation task doesnt have a default config, the config is based on the category. 

### Clinical Test

```
model_parameters:
  temperature: 0
  max_tokens: 1600

tests:
  defaults:
    min_pass_rate: 1.0

  clinical:
    demographic-bias:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

### Disinformation Test

```
model_parameters:
  max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  disinformation:
    narrative_wedging:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

### Security

```
model_parameters:
  temperature: 0.2
  max_tokens: 200

tests:
  defaults:
    min_pass_rate: 1.0

  security:
    prompt_injection_attack:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

### Toxicity

```
model_parameters:
  temperature: 0.2
  max_tokens: 200

tests:
  defaults:
    min_pass_rate: 1.0

  toxicity:
    offensive:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

## Translation

```
model_parameters:
  target_language: 'fr'
  
tests:
  defaults:
    min_pass_rate: 1.0
  robustness:
    add_typo:
      min_pass_rate: 0.70
    uppercase:
      min_pass_rate: 0.70
```

</div><div class="h3-box" markdown="1">

Note: If you are using Azure OpenAI, please ensure that you modify the Config file by adding the `deployment_name` parameter under the `model_parameters` section.
</div><div class="h3-box" markdown="1">

</div></div>