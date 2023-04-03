---
layout: docs
seotitle: One Liners | NLP Test | John Snow Labs
title: One Liners
permalink: /docs/pages/docs/one_liner
key: docs-install
modify_date: "2023-03-28"
header: true
---

With just one line of code, it can generate and run over 50 different test types to assess the quality of NLP models in terms of accuracy, bias, robustness, representation, and fairness. 

<div class="heading" id="ner">One Liner - NER</div>

```python
from nlptest import Harness
h = Harness(task='ner', model='ner.dl', hub='johnsnowlabs')

# Generate test cases, run them and view a report
h.generate().run().report()
```

<div class="heading" id="classification">One Liner - Text Classification </div>

```python
from nlptest import Harness
h = Harness(task='text-classification', model='en.sentiment.imdb.glove', hub='johnsnowlabs')

# Generate test cases, run them and view a report
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

  #ner {
    color: #1E77B7;
  }
  
  #classification {
    color: #1E77B7;
  }
  

</div></div>