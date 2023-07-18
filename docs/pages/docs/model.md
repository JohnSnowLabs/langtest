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
The `Harness` `model` parameter accepts either a pretrained model or pipeline from a given hub, a custom pipeline object, or a path to a locally saved model.

### John Snow Labs


#### Pretrained Models

```bash
!pip install johnsnowlabs
```

```python
from langtest import Harness
h = Harness(task='ner', model='ner_dl_bert', hub='johnsnowlabs', data='test.conll', config='config.yml')

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
h = Harness(task='ner', model=ner_model_pipeline, hub='johnsnowlabs', data='test.conll', config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

#### Locally Saved Models

```python
from langtest import Harness

# Create test Harness
h = Harness(task='ner', model='path/to/local_saved_model', hub='johnsnowlabs', data='test.conll', config='config.yml')

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
h = Harness(task='ner', model='dslim/bert-base-NER', hub='huggingface', data='test.conll', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

#### Locally Saved Models

```python
from langtest import Harness

# Create test Harness
h = Harness(task='text-classification', model='path/to/local_saved_model', hub='huggingface', data='test.csv', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

### OpenAI

Using any large language model from the [OpenAI API](https://platform.openai.com/docs/models/overview):

```bash
!pip install "langtest[langchain,openai,transformers]"
```

```python
from langtest import Harness

# Set API keys
os.environ['OPENAI_API_KEY'] = ''

# Create test Harness
h = Harness(task='question-answering', model='gpt-3.5-turbo', hub='openai', data='BoolQ-test', config='config.yml')

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
h = Harness(task='ner', model='en_core_web_sm', hub='spacy', data='test.conll', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

#### Locally Saved Models

```python
from langtest import Harness

# Create test Harness
h = Harness(task='text-classification', model='path/to/local_saved_model', hub='spacy', data='test.csv', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```
<div class="h3-box" markdown="1">

### Cohere

#### Pretrained Models

```bash
!pip install "langtest[transformers,langchain,cohere]"
```

```python
from langtest import Harness

# Set API keys
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"

# Create test Harness
h = Harness(task="question-answering", hub="cohere", model="command-xlarge-nightly", data='BoolQ-test', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>

### AI21

#### Pretrained Models

```bash
!pip install "langtest[transformers,langchain,ai21]"
```

```python
from langtest import Harness

# Set API keys
os.environ["AI21_API_KEY"] = "<YOUR_API_KEY>"

# Create test Harness
h = Harness(task="question-answering", hub="ai21", model="j2-jumbo-instruct", data='BoolQ-test-tiny', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```
<div class="h3-box" markdown="1">

### Azure OpenAI

#### Pretrained Models

```bash
!pip install "langtest[transformers,langchain,openai]"
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
h = Harness(task="question-answering", hub="azure-openai", model="text-davinci-003", data='BoolQ-test-tiny', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>
<div class="h3-box" markdown="1">

### Huggingface Inference-Api

#### Pretrained Models

```bash
!pip install "langtest[transformers,langchain,huggingface-hub]"
```

```python
from langtest import Harness

# Set API keys
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<API_TOKEN>"

# Create test Harness
h = Harness(task="question-answering", hub="huggingface-inference-api", model="google/flan-t5-small", data='BoolQ-test-tiny')
# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div>

</div></div>