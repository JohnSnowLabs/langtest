---
layout: landing
title: 'NLU: <span>State of the Art Text Mining in Python</span>'
excerpt:  <br> The Simplicity of Python, the Power of Spark NLP
permalink: /
header: true
article_header:
 actions:
   - text: Getting Started
     type: active
     url: /docs/en/install   
   - text: '<i class="fab fa-github"></i> GitHub'
     type: trans
     url: https://github.com/JohnSnowLabs/nlu 
   - text: '<i class="fab fa-slack-hash"></i> Slack'
     type: trans
     url: https://app.slack.com/client/T9BRVC9AT/C0196BQCDPY   

 height: 50vh
 theme: dark

data:
 sections:
   - title:
     children:
       - title: Powerful One-Liners
         image: 
            src: /assets/images/powerfull_one.svg
         excerpt: Over a thousand NLP models in hundreds of languages are at your fingertips with just one line of code
       - title: Elegant Python
         image: 
            src: /assets/images/elegant_python.svg
         excerpt: Directly read and write pandas dataframes for frictionless integration with other libraries and existing ML pipelines  
       - title: 100% Open Source
         image: 
            src: /assets/images/open_source.svg
         excerpt: Including pre-trained models & pipelines

   - title: 'Quick and Easy'
     install: yes
     excerpt: NLU is available on <a href="https://pypi.org/project/nlu" target="_blank">PyPI</a>, <a href="https://anaconda.org/JohnSnowLabs/nlu" target="_blank">Conda</a>
     actions:
       - text: Install NLU
         type: big_btn
         url: /docs/en/install
  
  
   - title: Benchmark
     excerpt: NLU is based on the award winning Spark NLP which best performing in peer-reviewed results
     benchmark: yes
     features: false
     theme: dark

   - title: Pandas
     excerpt: NLU ships with many <b>NLP features</b>, pre-trained <b>models</b> and <b>pipelines</b> <div>It takes in Pandas and outputs <b>Pandas Dataframes</b></div><div>All in <b>one line</b></div>
     pandas: yes
     theme: dark

    
---