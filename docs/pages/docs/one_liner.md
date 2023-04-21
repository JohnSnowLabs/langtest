---
layout: docs
seotitle: One Liners | NLP Test | John Snow Labs
title: One Liners
permalink: /docs/pages/docs/one_liner
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
With just one line of code, you can generate and run over 50 different test types to assess the quality of **John Snow Labs**, **Hugging Face**, and **Spacy** models. These tests fall into robustness, accuracy, bias, representation and fairness test categories for NER and Text Classification models, with support for many more models and test types coming soon.
</div> 

### One Liner - NER

Try out the nlptest library on the following default model/dataset combinations for NER. **To run tests on any other model, make sure to provide a dataset that matches your model's label predictions** (check [Test Harness docs](https://nlptest.org/docs/pages/docs/harness)).

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
from nlptest import Harness

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
from nlptest import Harness

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
from nlptest import Harness

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

Try out the nlptest library on the following default model/dataset combinations for Text Classification. **To run tests on any other model, make sure to provide a dataset that matches your model's label predictions** (check [Test Harness docs](https://nlptest.org/docs/pages/docs/harness)).

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
from nlptest import Harness

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
from nlptest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model='mrm8488/distilroberta-finetuned-tweets-hate-speech', hub='huggingface')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
from nlptest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model='textcat_imdb', hub='spacy')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

</div>