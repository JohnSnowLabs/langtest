---
layout: docs
header: true
seotitle: Tests | LangTest | John Snow Labs
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

The tables presented below offer a comprehensive overview of diverse categories and tests, providing valuable insights into the varied testing procedures.

## Accuracy Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Accuracy](accuracy)             | [Min F1 Score](accuracy#min-f1-score)                                                                           | `ner`, `text-classification`                                                        |
| [Accuracy](accuracy)             | [Min Macro-F1 Score](accuracy#min-macro-f1-score)                                                               | `ner`, `text-classification`                                                        |
| [Accuracy](accuracy)             | [Min Micro-F1 Score](accuracy#min-micro-f1-score)                                                               | `ner`, `text-classification`                                                        |
| [Accuracy](accuracy)             | [Min Precision Score](accuracy#min-precision-score)                                                             | `ner`, `text-classification`                                                        |
| [Accuracy](accuracy)             | [Min Recall Score](accuracy#min-recall-score)                                                                   | `ner`, `text-classification`                                                        |
| [Accuracy](accuracy)             | [Min Weighted-F1 Score](accuracy#min-weighted-f1-score)                                                         | `ner`, `text-classification`                                                        |
| [Accuracy](accuracy)             | [Min Exact Match Score](accuracy#min-exact=match-score)                                                         | `question-answering`, `summarization`                                               |
| [Accuracy](accuracy)             | [Min BLEU Score](accuracy#min-bleu-score)                                                                       | `question-answering`, `summarization`                                               |
| [Accuracy](accuracy)             | [Min Rouge1 Score](accuracy#min-rouge1-score)                                                                   | `question-answering`, `summarization`                                               |
| [Accuracy](accuracy)             | [Min Rouge2 Score](accuracy#min-rouge2-score)                                                                   | `question-answering`, `summarization`                                               |
| [Accuracy](accuracy)             | [Min RougeL Score](accuracy#min-rougel-score)                                                                   | `question-answering`, `summarization`                                               |
| [Accuracy](accuracy)             | [Min RougeLsum Score](accuracy#min-rougelsum-score)                                                             | `question-answering`, `summarization`                                               |

## Bias Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Bias](bias)                     | [Replace To Asian First Names](bias#replace-to-asian-firstnames)                                                | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Asian Last Names](bias#replace-to-asian-lastnames)                                                  | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Black First Names](bias#replace-to-black-firstnames)                                                | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Black Last Names](bias#replace-to-black-lastnames)                                                  | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Buddhist Names](bias#replace-to-buddhist-names)                                                     | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Christian Names](bias#replace-to-christian-names)                                                   | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Female Pronouns](bias#replace-to-female-pronouns)                                                   | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To High Income Country](bias#replace-to-high-income-country)                                           | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Hindu Names](bias#replace-to-hindu-names)                                                           | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Hispanic First Names](bias#replace-to-hispanic-firstnames)                                          | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Hispanic Last Names](bias#replace-to-hispanic-lastnames)                                            | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Interracial Last Names](bias#replace-to-inter-racial-lastnames)                                     | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Jain Names](bias#replace-to-jain-names)                                                             | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Lower Middle Income Country](bias#replace-to-lower-middle-income-country)                           | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Low Income Country](bias#replace-to-low-income-country)                                             | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Male Pronouns](bias#replace-to-male-pronouns)                                                       | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Muslim Names](bias#replace-to-muslim-names)                                                         | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Native American Last Names](bias#replace-to-native-american-lastnames)                              | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Neutral Pronouns](bias#replace-to-neutral-pronouns)                                                 | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Parsi Names](bias#replace-to-parsi-names)                                                           | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Sikh Names](bias#replace-to-sikh-names)                                                             | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To Upper Middle Income Country](bias#replace-to-upper-middle-income-country)                           | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To White First Names](bias#replace-to-white-firstnames)                                                | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Bias](bias)                     | [Replace To White Last Names](bias#replace-to-white-lastnames)                                                  | `ner`, `text-classification`, `question-answering`, `summarization`                 |

## Fairness Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Fairness](fairness)             | [Max Gender F1 Score](fairness#max-gender-f1-score)                                                             | `ner`, `text-classification`                                                        |
| [Fairness](fairness)             | [Min Gender F1 Score](fairness#min-gender-f1-score)                                                             | `ner`, `text-classification`                                                        |
| [Fairness](fairness)             | [Min Gender Rouge1 Score](fairness#min-gender-rouge1-score)                                                     | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Min Gender Rouge2 Score](fairness#min-gender-rouge2-score)                                                     | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Min Gender RougeL Score](fairness#min-gender-rougeL-score)                                                     | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Min Gender RougeLSum Score](fairness#min-gender-rougeLsum-score)                                               | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Max Gender Rouge1 Score](fairness#max-gender-rouge1-score)                                                     | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Max Gender Rouge2 Score](fairness#max-gender-rouge2-score)                                                     | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Max Gender RougeL Score](fairness#max-gender-rougeL-score)                                                     | `question-answering`, `summarization`                                               |
| [Fairness](fairness)             | [Max Gender RougeLSum Score](fairness#max-gender-rougeLsum-score)                                               | `question-answering`, `summarization`                                               |

## Representation Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Representation](representation) | [Min Country Economic Representation Count](representation#min-country-economic-representation-count)           | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Country Economic Representation Proportion](representation#min-country-economic-representation-proportion) | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Ethnicity Representation Count](representation#min-ethnicity-representation-count)                         | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Ethnicity Representation Proportion](representation#min-ethnicity-representation-proportion)               | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Gender Representation Count](representation#min-gender-representation-count)                               | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Gender Representation Proportion](representation#min-gender-representation-proportion)                     | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Label Representation Count](representation#min-label-representation-count)                                 | `ner`, `text-classification`                                                        |
| [Representation](representation) | [Min Label Representation Proportion](representation#min-label-representation-proportion)                       | `ner`, `text-classification`                                                        |
| [Representation](representation) | [Min Religion Name Representation Count](representation#min-religion-name-representation-count)                 | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Representation](representation) | [Min Religion Name Representation Proportion](representation#min-religion-name-representation-proportion)       | `ner`, `text-classification`, `question-answering`, `summarization`                 |

## Robustness Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Robustness](robustness)         | [Add Context](robustness#add-context)                                                                           | `ner`, `text-classification`, `question-answering`, `summarization` , `translation` |
| [Robustness](robustness)         | [Add Contraction](robustness#add-contraction)                                                                   | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Add Punctuation](robustness#add-punctuation)                                                                   | `ner`, `text-classification`, `question-answering`, `summarization` , `translation` |
| [Robustness](robustness)         | [Add Typo](robustness#add-typo)                                                                                 | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [American to British](robustness#american-to-british)                                                           | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [British to American](robustness#british-to-american)                                                           | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Lowercase](robustness#lowercase)                                                                               | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Strip Punctuation](robustness#strip-punctuation)                                                               | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Swap Entities](robustness#swap-entities)                                                                       | `ner`                                                                               |
| [Robustness](robustness)         | [Titlecase](robustness#titlecase)                                                                               | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Uppercase](robustness#uppercase)                                                                               | `ner`, `text-classification`, `question-answering`, `summarization` , `translation` |
| [Robustness](robustness)         | [Number to Word](robustness#number-to-word)                                                                     | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Add OCR Typo](robustness#add-ocr-typo)                                                                         | `ner`, `text-classification`, `question-answering`, `summarization`. `translation`  |
| [Robustness](robustness)         | [Dyslexia Word Swap](robustness#dyslexia-word-swap)                                                             | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Add Slangs](robustness#add-slangs)                                                                             | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Add Speech to Text Typo](robustness#add-speech-to-text-typo)                                                   | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Add Abbreviations](robustness#add-abbreviation)                                                                | `ner`, `text-classification`, `question-answering`, `summarization`                 |
| [Robustness](robustness)         | [Multiple Perturbations](robustness#multiple-perturbations)                                                     | `text-classification`, `question-answering`, `summarization`, `translation`         |
| [Robustness](robustness)         | [Adjective Synonym Swap](robustness#adjective-synonym-swap)                                                     | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Adjective Antonym Swap](robustness#adjective-antonym-swap)                                                     | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Strip All Punctution](robustness#strip-all-punctuation)                                                        | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |
| [Robustness](robustness)         | [Randomize Age](robustness#random-age)                                                                          | `ner`, `text-classification`, `question-answering`, `summarization`, `translation`  |

## Toxicity Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Toxicity](toxicity)             | [Offensive](toxicity#Offensive)                                                                                 | `text-generation`                                                                          |
| [Toxicity](toxicity)             | [ideology](toxicity#ideology)                                                                                   | `text-generation`                                                                          |
| [Toxicity](toxicity)             | [lgbtqphobia](toxicity#lgbtqphobia)                                                                             | `text-generation`                                                                          |
| [Toxicity](toxicity)             | [racism](toxicity#racism)                                                                                       | `text-generation`                                                                          |
| [Toxicity](toxicity)             | [sexism](toxicity#sexism)                                                                                       | `text-generation`                                                                          |
| [Toxicity](toxicity)             | [xenophobia](toxicity#xenophobia)                                                                               | `text-generation`                                                                          |

## Sensitivity Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Sensitivity](sensitivity)       | [Negation](sensitivity#negation)                                                                                | `question-answering`                                                                  |
| [Sensitivity](sensitivity)       | [Toxicity](sensitivity#toxicity)                                                                                | `question-answering`                                                                  |

## Sycophancy Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Sycophancy](sycophancy)       | [Sycophancy Math](sycophancy#sycophancy_math)                                                                                | `question-answering`                                                                  |
| [Sycophancy](sycophancy)       | [Sycophancy NLP](sycophancy#sycophancy_nlp)                                                                                | `question-answering`                                                                  |

## Stereotype Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Stereotype](stereotype)           | [gender-occupational-stereotype](stereotype#gender-occupational-stereotype)                                      | `fill-mask`                                                                         |
| [Stereotype](stereotype)       | [common-stereotypes](stereotype#common-stereotypes)                                                            | `fill-mask`                                                                       |

## StereoSet Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [StereoSet](stereoset)           | [intersentence](stereoset#intersentence)                                                                        | `question-answering`                                                                         |
| [StereoSet](stereoset)           | [intrasentence](stereoset#intrasentence)                                                                        | `question-answering`                                                                         |

## Ideology Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Ideology](ideology)           | [Political Compass](ideology#political_compass)                                                                        | `ideology`|

## Legal Tests

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Legal](Legal)                   | [legal-support](legal#legal-support)                                                                            | `question-answering`                                                                       |

## Clinical Test


{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Clinical](clinical)             | [demographic-bias](clinical#demographic-bias)                                                                   | `text-generation`                                                                    |

## Security Test

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Security](security)             | [prompt_injection_attack](security#prompt_injection_attack)                                                     | `text-generation`                                                                          |

## Disinformation Test

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Disinformation](disinformation) | [Narrative Wedging](disinformation#narrative_wedging)                                                           | `text-generation`                                                               |


## Factuality Test

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Factuality](factuality)         | [Order Bias](factuality#order_bias)                                                                             | `question-answering`                                                                   |                                                               


## Grammar Test

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Grammar](grammar)         | [Paraphrase](grammar#paraphrase)                                                                             | `text-classification`, `question-answering`                                                                   |                                                               

</div></div>