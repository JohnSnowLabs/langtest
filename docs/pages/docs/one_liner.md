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
  <p>With just one line of code, you can generate and run over 50 different test types to assess the quality of **John Snow Labs**, **Hugging Face**, and **Spacy** models. These tests fall into robustness, accuracy, bias, representation and fairness test categories for NER and Text Classification models, with support for many more tasks coming soon.</p>
</div> 

### One Liner - NER

<div class="tabs-wrapper h3-box">
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

# Create a Harness object
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

# Create a Harness object
h = Harness(task='ner', model='en_core_web_sm', hub='spacy')

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Text Classification

```python
from nlptest import Harness
h = Harness(task='text-classification', model='en.sentiment.imdb.glove', hub='johnsnowlabs')

# Generate test cases, run them and view a report
h.generate().run().report()
```

</div>