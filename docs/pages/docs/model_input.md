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
For model inputs, we can pass either a pretrained model/pipeline from the hub or our own custom pipeline model, or path to a locally saved model.

<div class="heading" id="model">John Snow Labs</div>

```python
pip install johnsnowlabs
from johnsnowlabs import nlp
```

Using a `Pretrained Model` in John Snow Labs.

```python
from nlptest import Harness
h = Harness(task='ner', model='ner_dl_bert', hub='johnsnowlabs')

# Generate test cases, run them and view a report
h.generate().run().report()
```

Using a `Custom Pipeline` in John Snow Labs.

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

ner_pipeline = nlp.Pipeline().setStages([
				documentAssembler,
				tokenizer,
				embeddings,
				ner
    ])

ner_model_pipeline = ner_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))


from nlptest import Harness
h = Harness(task='ner', model=ner_model_pipeline, data='test.conll', config='test.config')
h.generate().run().report()


```

Using a `Locally Saved Model` in John Snow Labs .

```python
from nlptest import Harness
h = Harness(task='ner', model='path/to/local_saved_model', hub='johnsnowlabs', data='test.conll', config='test.config')

# Generate test cases, run them and view a report
h.generate().run().report()
```



<div class="heading" id="model">Spacy</div>

Using a `Pretrained Model` in Spacy.
```python
from nlptest import Harness

# Create a Harness object
h = Harness('ner', model='en_core_web_sm', hub='spacy')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

<div class="heading" id="model">Hugging Face</div>

Using a `Pretrained Model` in Hugging Face.
```python
from nlptest import Harness

# Create a Harness object
h = Harness('ner', model='dslim/bert-base-NER', hub='huggingface')

# Generate, run and get a report on your test cases
h.generate().run().report()
```

<style>
  .heading {
    text-align: center;
    font-size: 26px;
    font-weight: 500;
    padding-top: 20px;
    padding-bottom: 20px;
  }

  #model {
    color: #1E77B7;
  }

</div></div>