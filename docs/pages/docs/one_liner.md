---
layout: docs
seotitle: One Liners | LangTest | John Snow Labs
title: One Liners
permalink: /docs/pages/docs/one_liner
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
With just one line of code, you can generate and run over 50 different test types to assess the quality of **John Snow Labs**, **Hugging Face**, **OpenAI** and **Spacy** models. These tests fall into robustness, accuracy, bias, representation and fairness test categories for NER, Text Classification and Question Answering models, with support for many more models and test types actively being developed.
</div> 

### One Liner - NER

Try out the LangTest library on the following default model-dataset combinations for NER. **To run tests on any model other than those displayed in the code snippets here, make sure to provide a dataset that matches your model's label predictions** (check [Test Harness docs](https://langtest.org/docs/pages/docs/harness)).

<div id="one_liner_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">John Snow Labs</a>
    <a href="#" class="tab-btn">Hugging Face</a>
    <a href="#" class="tab-btn">Spacy</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='ner', model='ner.dl', hub='johnsnowlabs')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='ner', model='dslim/bert-base-NER', hub='huggingface')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='ner', model='en_core_web_sm', hub='spacy')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Text Classification

Try out the LangTest library on the following default model-dataset combinations for Text Classification. **To run tests on any model other than those displayed in the code snippets here, make sure to provide a dataset that matches your model's label predictions** (check [Test Harness docs](https://langtest.org/docs/pages/docs/harness)).

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">John Snow Labs</a>
    <a href="#" class="tab-btn">Hugging Face</a>
    <a href="#" class="tab-btn">Spacy</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model='en.sentiment.imdb.glove', hub='johnsnowlabs')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model='lvwerra/distilbert-imdb', hub='huggingface')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model='textcat_imdb', hub='spacy')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>


### One Liner - Question Answering

Try out the LangTest library on the following default model-dataset combinations for Question Answering. To get a list of valid dataset options, please navigate to the [Data Input docs](https://langtest.org/docs/pages/docs/data).

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">OpenAI</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create a Harness object
h = Harness(task='question-answering', model='gpt-3.5-turbo', hub='openai', data='BoolQ-test')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>


### One Liner - Summarization

Try out the LangTest library on the following default model-dataset combinations for Summarization. To get a list of valid dataset options, please navigate to the [Data Input docs](https://langtest.org/docs/pages/docs/data).

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">OpenAI</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create a Harness object
h = Harness(task='summarization', model='text-davinci-002', hub='openai', data='XSum-test-tiny')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Toxicity

Try out the LangTest library on the following default model-dataset combinations for Toxicity. To get a list of valid dataset options, please navigate to the [Data Input docs](https://langtest.org/docs/pages/docs/data).

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-header">
    <a href="#" class="tab-btn">OpenAI</a>
  </div>
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create a Harness object
h = Harness(task='toxicity', model='text-davinci-002', hub='openai', data='toxicity-test-tiny')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

</div>


### One Liner - Model Comparisons

To compare different models (either from same or different hubs) on the same task and test configuration, you can pass a dictionary to the 'model' parameter of the harness. This dictionary should contain the names of the models you want to compare, each paired with its respective hub.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from langtest import Harness

# Define the dictionary
model_comparison_dict = {
    "ner.dl":"johnsnowlabs",
    "dslim/bert-base-NER":"huggingface",
    "en_core_web_sm":"spacy"
}

# Create a Harness object
harness = Harness(task='ner', model=model_comparison_dict, data="/path-to-test-conll")

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>
