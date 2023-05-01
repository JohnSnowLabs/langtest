---
layout: docs
seotitle: Model Input | NLP Test | John Snow Labs
title: Model Input
permalink: /docs/pages/docs/model_input
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
The `Harness` `model` parameter accepts either a pretrained model or pipeline from a given hub, a custom pipeline object, or a path to a locally saved model.

### John Snow Labs


#### Pretrained Models

```bash
pip install johnsnowlabs
```

```python
from johnsnowlabs import nlp
```

```python
from nlptest import Harness
h = Harness(task='ner', model='ner_dl_bert', hub='johnsnowlabs', data='test.conll', config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

#### Custom Pipelines

```python
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

from nlptest import Harness

# Create test Harness
h = Harness(task='ner', model=ner_model_pipeline, hub='johnsnowlabs', data='test.conll', config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

#### Locally Saved Models

```python
from nlptest import Harness

# Create test Harness
h = Harness(task='ner', model='path/to/local_saved_model', hub='johnsnowlabs', data='test.conll', config='config.yml')

# Generate test cases, run them and view a report
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

### Hugging Face

#### Pretrained Models

```python
from nlptest import Harness

# Create test Harness
h = Harness(task='ner', model='dslim/bert-base-NER', hub='huggingface', data='test.conll', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

#### Locally Saved Models

```python
from nlptest import Harness

# Create test Harness
h = Harness(task='text-classification', model='path/to/local_saved_model', hub='huggingface', data='test.csv', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div><div class="h3-box" markdown="1">

### OpenAI

Using any large language model from the [OpenAI API](https://platform.openai.com/docs/models/overview):

```python
from nlptest import Harness

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

```python
from nlptest import Harness

# Create test Harness
h = Harness(task='ner', model='en_core_web_sm', hub='spacy', data='test.conll', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

#### Locally Saved Models

```python
from nlptest import Harness

# Create test Harness
h = Harness(task='text-classification', model='path/to/local_saved_model', hub='spacy', data='test.csv', config='config.yml')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

</div></div>