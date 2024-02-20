---
layout: docs
seotitle: Model | LangTest | John Snow Labs
title: Model
permalink: /docs/pages/docs/model
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


**model**: `Union[dict, list]`

Specifies the model to be evaluated. This parameter can be provided as either a dictionary or a list of dictionaries. Each dictionary should contain the following keys:

- model (mandatory): PipelineModel or path to a saved model or pretrained pipeline/model from hub.

- hub (mandatory): Hub (library) to use in back-end for loading model from public models hub or from path

#### Supported Hubs

example:-

- model: `dict`

```python
model= {"model" : "bert-base-uncased", "hub":"huggingface" }
```

- model: `list`

```python
model = [
    {"model": "ner.dl", "hub": "johnsnowlabs"},
    {"model": "en_core_web_sm", "hub": "spacy"},
]
```

To get more information about the supported hub, click [here](/docs/pages/docs/hub).

</div><div class="h3-box" markdown="1">

### John Snow Labs


#### Pretrained Models

```bash
!pip install johnsnowlabs
```

```python
from langtest import Harness
h = Harness(task='ner', model={'model': 'ner.dl', 'hub':'johnsnowlabs'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

#### Custom Pipelines

```python
from johnsnowlabs import nlp
spark = nlp.start()

documentAssembler = nlp.DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

tokenizer = nlp.Tokenizer()\
		.setInputCols(["document"])\
		.setOutputCol("token")
	
embeddings = nlp.WordEmbeddingsModel.pretrained('glove_100d') \
		.setInputCols(["document", 'token']) \
		.setOutputCol("embeddings")

ner = nlp.NerDLModel.pretrained("ner_dl", 'en') \
		.setInputCols(["document", "token", "embeddings"]) \
		.setOutputCol("ner")

ner_pipeline = nlp.Pipeline().setStages([documentAssembler,
                                         tokenizer,
                                         embeddings,
                                         ner])

ner_model_pipeline = ner_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

from langtest import Harness

# Create test Harness
h = Harness(task='ner', model={'model': ner_model_pipeline, 'hub':'johnsnowlabs'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

#### Locally Saved Models

```python
from langtest import Harness

# Create test Harness
h = Harness(task='ner', model={'model': 'path/to/local_saved_model', 'hub':'johnsnowlabs'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

### Hugging Face

#### Pretrained Models

```bash
!pip install langtest[transformers]
```

```python
from langtest import Harness

# Create test Harness
h = Harness(task='ner', model={'model': 'dslim/bert-base-NER', 'hub':'huggingface'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

#### Locally Saved Models

```python
from langtest import Harness

# Create test Harness
h = Harness(task='text-classification', model={'model': 'path/to/local_saved_model', 'hub':'huggingface'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

### OpenAI

Using any large language model from the [OpenAI API](https://platform.openai.com/docs/models/overview):

```bash
!pip install "langtest[openai]"
```

```python
from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create test Harness
h = Harness(task="question-answering", 
            model={"model": "gpt-3.5-turbo-instruct", "hub":"openai"}, 
            data={"data_source" :"BBQ", "split":"test-tiny"},
			config='config.yml')
# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

### Spacy

#### Pretrained Models

```bash
!pip install spacy
```

```python
from langtest import Harness

# Create test Harness
h = Harness(task='ner', model={'model': 'en_core_web_sm', 'hub':'spacy'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

#### Locally Saved Models

```python
from langtest import Harness

# Create test Harness
h = Harness(task='text-classification', model={'model': 'path/to/local_saved_model', 'hub':'spacy'}, data={'data_source':'test.conll'}, config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```
<div class="h3-box" markdown="1">

### Cohere

#### Pretrained Models

```bash
!pip install "langtest[langchain,cohere]"
```

```python
from langtest import Harness

# Set API keys
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"

# Create test Harness
h = Harness(task="question-answering", 
			model={"model": "command-xlarge-nightly", "hub":"cohere"}, 
			data={"data_source" :"NQ-open", "split":"test-tiny"},
			config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>

### AI21

#### Pretrained Models

```bash
!pip install "langtest[langchain,ai21]"
```

```python
from langtest import Harness

# Set API keys
os.environ["AI21_API_KEY"] = "<YOUR_API_KEY>"

# Create test Harness
h = Harness(task="question-answering", 
            model={"model": "j2-jumbo-instruct", "hub":"ai21"}, 
            data={"data_source" :"BBQ", "split":"test-tiny"},
			config='config.yml')


# Generate, run and get a report on your test cases
h.generate().run().report()
```
<div class="h3-box" markdown="1">

### Azure OpenAI

#### Pretrained Models

```bash
!pip install "langtest[openai]"
```

```python
from langtest import Harness

# Set API keys
os.environ["OPENAI_API_KEY"] = "<API_KEY>"
openai.api_type = "azure"
openai.api_base = "<ENDPOINT>"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create test Harness
h = Harness(task="question-answering", 
            model={"model": "gpt-3.5-turbo-instruct", 'hub':'azure-openai'}, 
            data={"data_source" :"BBQ", "split":"test-tiny"},
			config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>
<div class="h3-box" markdown="1">

### Huggingface Inference-Api

#### Pretrained Models

```bash
!pip install "langtest[langchain,huggingface-hub]"
```

```python
from langtest import Harness

# Set API keys
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<API_TOKEN>"

# Create test Harness
h = Harness(task="question-answering", 
            model={"model": "google/flan-t5-small","hub": "huggingface-inference-api"},
            data={"data_source" :"BoolQ", "split":"test-tiny"},
            config='config.yml')
			
# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>

<div class="h3-box" markdown="1">

### LM Studio

#### Pretrained Models

```bash
!pip install langtest requests
```

```python
from langtest import Harness

# Create test Harness
h = Harness(task="question-answering", 
            model={"model": "http://localhost:1234/v1/chat/completions", "hub": "lm-studio"},
            data={"data_source" :"BoolQ", "split":"test-tiny"},
            config='config.yml')
			
# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>

<div class="h3-box" markdown="1">

### Custom

#### Custom model
```bash
!pip install "langtest"
```

```python
from langtest import Harness

# Create test Harness
harness = Harness(task="text-classification",
                  model={'model': model, "hub": "custom"}, 
                  data={'data_source': 'test.csv'},
                  config='config.yml')	

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>

</div></div>