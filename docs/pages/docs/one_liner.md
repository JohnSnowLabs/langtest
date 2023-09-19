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
!pip install "langtest[johnsnowlabs,transformers]"

from langtest import Harness
# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='ner', model={'model': 'ner.dl', 'hub':'johnsnowlabs'})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[transformers]

from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='ner', model={'model': 'dslim/bert-base-NER', 'hub':'huggingface'})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[spacy,transformers]" 

from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='ner', model={'model': 'en_core_web_sm', 'hub':'spacy'})

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
!pip install "langtest[johnsnowlabs,transformers]"

from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model={'model': 'en.sentiment.imdb.glove', 'hub':'johnsnowlabs'})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[transformers]

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model={'model': 'lvwerra/distilbert-imdb', 'hub':'huggingface'})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[spacy,transformers]"

from langtest import Harness

# Make sure to specify data='path_to_data' when using custom models
h = Harness(task='text-classification', model={'model': 'textcat_imdb', 'hub':'spacy'})

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
!pip install "langtest[langchain,openai,transformers]"

from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create a Harness object
h = Harness(task="question-answering", 
            model={"model": "text-davinci-003","hub":"openai"}, 
            data={"data_source" :"BoolQ-test"})

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
!pip install "langtest[evaluate,langchain,openai,transformers]"

from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create a Harness object
h = Harness(task="summarization",
            model={"model": "text-davinci-002","hub":"openai"}, 
            data={"data_source" :"XSum-test-tiny"})

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
!pip install "langtest[evaluate,langchain,openai,transformers]"

from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create a Harness object
h = Harness(task="toxicity", 
            model={"model": "text-davinci-002","hub":"openai"}, 
            data={"data_source" :"toxicity-test-tiny"})

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
!pip install "langtest[spacy,johnsnowlabs,transformers]" 
from langtest import Harness

# Define the list
models = [{"model": "ner.dl" , "hub":"johnsnowlabs"} , {"model":"en_core_web_sm", "hub": "spacy"}]

# Create a Harness object
h = Harness(task="ner", model=models, data={"data_source":'/path-to-test-conll'})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Translation

Try out the LangTest library on the following default model-dataset combinations for Translation.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[transformers]

from langtest import Harness

# Create a Harness object

h = Harness(task="translation",
                  model={"model":'t5-base', "hub": "huggingface"},
                  data={"data_source": "Translation-test"}
                  )
# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Clinical-Tests

Try out the LangTest library on the following default model-dataset combinations for Clinical-Tests.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[langchain,openai,transformers]"

import os
os.environ["OPENAI_API_KEY"] = <ADD OPEN-AI-KEY>

from langtest import Harness

# Create a Harness object
h = Harness(task="clinical-tests",
            model={"model": "text-davinci-003", "hub": "openai"},
            data = {"data_source": "Gastroenterology-files"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>



### One Liner - Security-Test

Try out the LangTest library on the following default model-dataset combinations for Security Test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[openai]

import os
os.environ["OPENAI_API_KEY"] = <ADD OPEN-AI-KEY>

from langtest import Harness

# Create a Harness object
h = Harness(task="security",
            model={'model': "text-davinci-003", "hub": "openai"},
            data={'data_source':'Prompt-Injection-Attack'})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>



### One Liner - Disinformation-Test

Try out the LangTest library on the following default model-dataset combinations for Disinformation-Test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[ai21,langchain,transformers]" 

import os
os.environ["AI21_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h  =  Harness(task="disinformation-test",
              model={"model": "j2-jumbo-instruct", "hub":"ai21"},
              data = {"data_source": "Narrative-Wedging"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Political-Test

Try out the LangTest library on the following default model for Political Test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[openai]

import os
os.environ["OPENAI_API_KEY"] = <ADD OPEN-AI-KEY>

from langtest import Harness

# Create a Harness object
h = Harness(task="political", model={'model': "text-davinci-003", "hub": "openai"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Factuality Test

Try out the LangTest library on the following default model-dataset combinations for Factuality Test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[openai,langchain,transformers]" 

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h  =  Harness(task="factuality-test",
              model = {"model": "text-davinci-003", "hub":"openai"},
              data = {"data_source": "Factual-Summary-Pairs"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Sensitivity Test

Try out the LangTest library on the following default model-dataset combinations for Sensitivity Test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[openai,langchain,transformers]" 

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h  =  Harness(task="sensitivity-test",
              model = {"model": "text-davinci-003", "hub":"openai"},
              data = {"data_source": "NQ-open-test-tiny"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Wino Bias

Try out the LangTest library on the following default model-dataset combinations for wino-bias test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[transformers]
!wget https://raw.githubusercontent.com/JohnSnowLabs/langtest/main/langtest/data/config/wino_config.yml

from langtest import Harness

# Create a Harness object
h = Harness(task="wino-bias", model={"model" : "bert-base-uncased", 
  "hub":"huggingface" } , data = {"data_source":"Wino-test"}, config="wino_config.yml")

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>


### One Liner - Legal Test

Try out the LangTest library on the following default model-dataset combinations for legal tests.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[openai,langchain]" 

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h = Harness(task="legal-tests", model={"model" : "text-davinci-002",
           "hub":"openai"}, data = {"data_source":"Legal-Support-test"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>