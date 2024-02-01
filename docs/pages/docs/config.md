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

## Configuring Tests

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


#### Understanding Model Parameters

When configuring Language Model (LLMs), certain parameters play a crucial role in determining the output quality and behavior of the model. Among these parameters, two are particularly important:

- **Temperature:** Controls the randomness of the generated text. Lower values result in more deterministic outputs, while higher values lead to more diverse but potentially less coherent text.

- **Max Tokens (or Max Length):** Specifies the maximum number of tokens (words or subwords) in the generated text. This parameter helps control the length of the output.

- **User prompt:** Users can provide a prompt that serves as a starting point  for the generated text. The prompt influences the content, style, and coherence of the generated text by guiding the model's understanding and focus.

- **Deployment Name**: With Azure OpenAI, you set up your own deployments of the common GPT-3 and Codex models. When calling the API, you need to specify the deployment you want to use. (Only applicable to Azure OpenAI Hub)

- **Stream**: Enables real-time partial response transmission during API interactions. (Only applicable to LM-Studio Hub)

- **Server Prompt**: Instructions or guidelines for the model to follow during the conversation. (Only applicable to LM-Studio Hub)

#### Model Parameter Variation Across Different Hubs

Different language model hubs may use slightly different parameter names or have variations in their default values. Here's a comparison across some popular hubs:

{:.table2}
| Hub Name                    | Parameter Name | 
|-----------------------------|----------------|
| AI21, Cohere, OpenAI, Hugging Face Inference API        | `temperature`, `max_tokens`, `user_promt`  |
| Azure OpenAI                | `temperature`, `max_tokens`, `user_promt`, `deployment_name` |
| LM-Studio                | `temperature`, `max_tokens`, `user_promt`, `stream`, `server_promt` |

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

> Please note that this represents the default configuration for Question-Answering. Below, you'll find the configurations for the categories falling under the Question-Answering task.

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

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">Negation</a>
    <a href="#" class="tab-btn">Toxicity</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
model_parameters:
  max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  sensitivity:
    negation:
      min_pass_rate: 0.70
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
model_parameters:
  max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  sensitivity:
    toxicity:
      min_pass_rate: 0.70
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

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

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">Sycophancy Math</a>
    <a href="#" class="tab-btn">Sycophancy NLP</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
model_parameters:
    max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  sycophancy:
    sycophancy_math:
      min_pass_rate: 0.70
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
model_parameters:
    max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  sycophancy:
    sycophancy_nlp:
      min_pass_rate: 0.70
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

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

The fill mask task doesn't come with a preset configuration; rather, the configuration varies depending on the category. Take a look below to see the configurations for different categories.

### Stereotype

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">Wino-Bias</a>
    <a href="#" class="tab-btn">CrowS-Pairs</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
model_parameters:
    max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  stereotype:
    wino-bias:
      min_pass_rate: 0.70
      diff_threshold: 0.03

{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
model_parameters:
    max_tokens: 64

tests:
  defaults:
    min_pass_rate: 1.0

  stereotype:
    crows-pairs:
      min_pass_rate: 0.70
      diff_threshold: 0.10
      filter_threshold: 0.15
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

</div><div class="h3-box" markdown="1">


## Config for Text-generation

The Text-generation task doesn't come with a preset configuration; rather, the configuration varies depending on the category. Take a look below to see the configurations for different categories.

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


</div></div>