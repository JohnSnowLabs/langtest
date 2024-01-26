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
!pip install langtest[johnsnowlabs]

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
!pip install langtest[spacy]

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
!pip install langtest[johnsnowlabs]

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

from langtest import Harness

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
!pip install langtest[spacy]

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
!pip install "langtest[openai]"

from langtest import Harness

# Set API keys
import os
os.environ['OPENAI_API_KEY'] = "<ADD OPEN-AI-KEY>"

# Create a Harness object
h = Harness(task="question-answering", 
            model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
            data={"data_source" :"BBQ", "split":"test-tiny"})

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
!pip install "langtest[evaluate,openai,transformers]"

from langtest import Harness

# Set API keys
import os
os.environ['OPENAI_API_KEY'] = "<ADD OPEN-AI-KEY>"

# Create a Harness object
h = Harness(task="summarization", 
            model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
            data={"data_source" :"XSum", "split":"test-tiny"})

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
!pip install "langtest[evaluate,openai,transformers]"

from langtest import Harness

# Set API keys
import os
os.environ['OPENAI_API_KEY'] = "<ADD OPEN-AI-KEY>"

# Create a Harness object
h = Harness(task={"task":"text-generation", "category":"toxicity"}, 
            model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
            data={"data_source" :'Toxicity', "split":"test"})

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
!pip install "langtest[spacy,johnsnowlabs]" 
!wget https://raw.githubusercontent.com/JohnSnowLabs/langtest/main/langtest/data/conll/sample.conll

from langtest import Harness

# Define the list
models = [{"model": "ner.dl" , "hub":"johnsnowlabs"} , {"model":"en_core_web_sm", "hub": "spacy"}]

# Create a Harness object
h = Harness(task="ner", model=models, data={"data_source":'sample.conll'})

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
            data={"data_source": "Translation", "split":"test"})

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
!pip install "langtest[openai,transformers]"

import os
os.environ["OPENAI_API_KEY"] = "<ADD OPEN-AI-KEY>"

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"text-generation", "category":"clinical-tests"}, 
            model={"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
            data = {"data_source": "Clinical", "split":"Medical-files"})

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
os.environ["OPENAI_API_KEY"] = "<ADD OPEN-AI-KEY>"

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"text-generation", "category":"security"}, 
            model={'model': "gpt-3.5-turbo-instruct", "hub": "openai"},
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
h  =  Harness(task={"task":"text-generation", "category":"disinformation-test"}, 
              model={"model": "j2-jumbo-instruct", "hub":"ai21"},
              data = {"data_source": "Narrative-Wedging"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Ideology

Try out the LangTest library on the following default model for Ideology Test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[openai]

import os
os.environ["OPENAI_API_KEY"] = "<ADD OPEN-AI-KEY>"

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"question-answering", "category":"ideology"}, 
            model={'model': "gpt-3.5-turbo-instruct", "hub": "openai"})

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
!pip install "langtest[openai,transformers]" 

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h  =  Harness(task={"task":"question-answering", "category":"factuality-test"}, 
              model = {"model": "gpt-3.5-turbo-instruct", "hub":"openai"},
              data = {"data_source": "Factual-Summary-Pairs", "split":"test"})

              

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
! pip install "langtest[openai,transformers]"

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h  =  Harness(task={"task":"question-answering", "category":"sensitivity-test"}, 
              model = {"model": "gpt-3.5-turbo-instruct", "hub":"openai"},
              data={"data_source" :"NQ-open","split":"test-tiny"})

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

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"fill-mask", "category":"wino-bias"}, 
            model={"model" : "bert-base-uncased", "hub":"huggingface" }, 
            data ={"data_source":"Wino-test", "split":"test"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Wino Bias LLMs

Try out the LangTest library on the following default model-dataset combinations for wino-bias test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[openai]
from langtest import Harness

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# Create a Harness object
h=  Harness(task={"task":"question-answering", "category":"wino-bias"},
            model={"model": "gpt-3.5-turbo-instruct","hub":"openai"},
            data ={"data_source":"Wino-test", "split":"test"})

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
!pip install "langtest[openai]" 

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"question-answering", "category":"legal-tests"}, 
            model={"model" : "gpt-3.5-turbo-instruct", "hub":"openai"}, 
            data = {"data_source":"Legal-Support"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - Sycophancy Test

Try out the LangTest library on the following default model-dataset combinations for sycophancy test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install "langtest[openai]" 

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"question-answering", "category":"sycophancy-test"},
            model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
            data={"data_source": 'synthetic-math-data',})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>


### One Liner - Crows Pairs

Try out the LangTest library on the following default model-dataset combinations for crows-pairs test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[transformers]

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"fill-mask", "category":"crows-pairs"}, 
            model={"model" : "bert-base-uncased", "hub":"huggingface" },
            data = {"data_source":"Crows-Pairs"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

### One Liner - StereoSet

Try out the LangTest library on the following default model-dataset combinations for StereoSet test.

<div id="one_liner_text_tab" class="tabs-wrapper h3-box">
  <div class="tabs-body">
    <div class="tabs-item">
      <div class="highlight-box">
        {% highlight python %}
!pip install langtest[transformers]

from langtest import Harness

# Create a Harness object
h = Harness(task={"task":"question-answering", "category":"stereoset"}, 
            model={"model" : "bert-base-uncased", "hub":"huggingface" },
            data = {"data_source":"StereoSet"})

# Generate, run and get a report on your test cases
h.generate().run().report()
{% endhighlight %}
      </div>
    </div>
  </div>
</div>