---
layout: docs
header: true
seotitle: Tests | NLP Test | John Snow Labs
title: Available Tests
key: notebooks
permalink: /docs/pages/tests/test
aside:
    toc: true
sidebar:
    nav: tests
show_edit_on_github: true
nav_key: tests
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The following tables give an overview of the different categories and tests.

</div><div class="h3-box" markdown="1">

{:.table-model-big}
|Test Category|Test Name|Suppoted Tasks|
|-|
|[Accuracy](accuracy)		                |[Min F1 Score](accuracy#min-f1-score)                                                                      |`ner`, `text-classification`     
|[Accuracy](accuracy)		                |[Min Macro-F1 Score](accuracy#min-macro-f1-score)                                                          |`ner`, `text-classification`     
|[Accuracy](accuracy)		                |[Min Micro-F1 Score](accuracy#min-micro-f1-score)                                                          |`ner`, `text-classification`     
|[Accuracy](accuracy)		                |[Min Precision Score](accuracy#min-precision-score)                                                        |`ner`, `text-classification`       
|[Accuracy](accuracy)		                |[Min Recall Score](accuracy#min-recall-score)                                                              |`ner`, `text-classification`     
|[Accuracy](accuracy)		                |[Min Weighted-F1 Score](accuracy#min-weighted-f1-score)                                                    |`ner`, `text-classification`
|[Accuracy](accuracy)		                |[Min Exact Match Score](accuracy#min-exact=match-score)                                                    |`question-answering`
|[Accuracy](accuracy)		                |[Min BLEU Score](accuracy#min-bleu-score)                                                                  |`question-answering`
|[Accuracy](accuracy)		                |[Min Rouge1 Score](accuracy#min-rouge1-score)                                                              |`question-answering`
|[Accuracy](accuracy)		                |[Min Rouge2 Score](accuracy#min-rouge2-score)                                                              |`question-answering`
|[Accuracy](accuracy)		                |[Min RougeL Score](accuracy#min-rougel-score)                                                              |`question-answering`
|[Accuracy](accuracy)		                |[Min RougeLsum Score](accuracy#min-rougelsum-score)                                                        |`question-answering`       
|[Bias](bias)		                        |[Replace To Asian First Names](bias#replace-to-asian-firstnames)                                           |`ner`, `text-classification`      
|[Bias](bias)		                        |[Replace To Asian Last Names](bias#replace-to-asian-lastnames)                                             |`ner`, `text-classification`        
|[Bias](bias)		                        |[Replace To Black First Names](bias#replace-to-black-firstnames)                                           |`ner`, `text-classification`      
|[Bias](bias)		                        |[Replace To Black Last Names](bias#replace-to-black-lastnames)                                             |`ner`, `text-classification`        
|[Bias](bias)		                        |[Replace To Buddhist Names](bias#replace-to-buddhist-names)                                                |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Christian Names](bias#replace-to-christian-names)                                              |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To Female Pronouns](bias#replace-to-female-pronouns)                                              |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To High Income Country](bias#replace-to-high-income-country)                                      |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To Hindu Names](bias#replace-to-hindu-names)                                                      |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To Hispanic First Names](bias#replace-to-hispanic-firstnames)                                     |`ner`, `text-classification`        
|[Bias](bias)		                        |[Replace To Hispanic Last Names](bias#replace-to-hispanic-lastnames)                                       |`ner`, `text-classification`      
|[Bias](bias)		                        |[Replace To Interracial Last Names](bias#replace-to-inter-racial-lastnames)                                |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Jain Names](bias#replace-to-jain-names)                                                        |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Lower Middle Income Country](bias#replace-to-lower-middle-income-country)                      |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To Low Income Country](bias#replace-to-low-income-country)                                        |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Male Pronouns](bias#replace-to-male-pronouns)                                                  |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To Muslim Names](bias#replace-to-muslim-names)                                                    |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Native American Last Names](bias#replace-to-native-american-lastnames)                         |`ner`, `text-classification`        
|[Bias](bias)		                        |[Replace To Neutral Pronouns](bias#replace-to-neutral-pronouns)                                            |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Parsi Names](bias#replace-to-parsi-names)                                                      |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To Sikh Names](bias#replace-to-sikh-names)                                                        |`ner`, `text-classification`       
|[Bias](bias)		                        |[Replace To Upper Middle Income Country](bias#replace-to-upper-middle-income-country)                      |`ner`, `text-classification`     
|[Bias](bias)		                        |[Replace To White First Names](bias#replace-to-white-firstnames)                                           |`ner`, `text-classification`      
|[Bias](bias)		                        |[Replace To White Last Names](bias#replace-to-white-lastnames)                                             |`ner`, `text-classification`        
|[Fairness](fairness)		                |[Max Gender F1 Score](fairness#max-gender-f1-score)                                                        |`ner`, `text-classification`       
|[Fairness](fairness)		                |[Min Gender F1 Score](fairness#min-gender-f1-score)                                                        |`ner`, `text-classification`       
|[Representation](representation)		    |[Min Country Economic Representation Count](representation#country-economic-representation-count)          |`ner`, `text-classification`, `question-answering`       
|[Representation](representation)		    |[Min Country Economic Representation Proportion](representation#country-economic-representation-proportion)|`ner`, `text-classification`, `question-answering`         
|[Representation](representation)		    |[Min Ethnicity Representation Count](representation#ethnicity-representation-count)                        |`ner`, `text-classification`, `question-answering`         
|[Representation](representation)		    |[Min Ethnicity Representation Proportion](representation#ethnicity-representation-proportion)              |`ner`, `text-classification`, `question-answering`       
|[Representation](representation)		    |[Min Gender Representation Count](representation#gender-representation-count)                              |`ner`, `text-classification`, `question-answering`       
|[Representation](representation)		    |[Min Gender Representation Proportion](representation#gender-representation-proportion)                    |`ner`, `text-classification`, `question-answering`         
|[Representation](representation)		    |[Min Label Representation Count](representation#label-representation-count)                                |`ner`, `text-classification`, `question-answering`         
|[Representation](representation)		    |[Min Label Representation Proportion](representation#label-representation-proportion)                      |`ner`, `text-classification`, `question-answering`       
|[Representation](representation)		    |[Min Gender Representation Count](representation#religion-representation-count)                            |`ner`, `text-classification`, `question-answering`         
|[Representation](representation)		    |[Min Gender Representation Proportion](representation#religion-representation-proportion)                  |`ner`, `text-classification`, `question-answering`       
|[Robustness](robustness)		            |[Add Context](robustness#add-context)                                                                      |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Add Contraction](robustness#add-contraction)                                                              |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Add Punctuation](robustness#add-punctuation)                                                              |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Add Typo](robustness#add-typo)                                                                            |`ner`, `text-classification`, `question-answering`       
|[Robustness](robustness)		            |[American to British](robustness#american-to-british)                                                      |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[British to American](robustness#british-to-american)                                                      |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Lowercase](robustness#lowercase)                                                                          |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Strip Punctuation](robustness#strip-punctuation)                                                          |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Swap Entities](robustness#swap-entities)                                                                  |`ner`     
|[Robustness](robustness)		            |[Titlecase](robustness#titlecase)                                                                          |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Uppercase](robustness#uppercase)                                                                          |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Number to Word](robustness#number-to-word)                                                                |`ner`, `text-classification`, `question-answering`     
|[Robustness](robustness)		            |[Add OCR Typo](robustness#add-ocr-typo)                                                                |`ner`, `text-classification`, `question-answering`
</div></div>
