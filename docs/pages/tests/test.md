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

Accuracy testing is vital for evaluating a machine learning model's performance. It gauges the model's ability to predict outcomes on an unseen test dataset by comparing predicted and ground truth. Several tests, including labelwise metrics (precision, recall, F1 score) and overall metrics (micro F1, macro F1, weighted F1), support this assessment.

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

The primary goal of model bias tests is to assess how well a model aligns its predictions with actual outcomes. Model bias, the systematic skewing of results, can lead to negative consequences such as perpetuating stereotypes or discrimination. In this context, the objective is to explore how replacing documents with different genders, ethnicities, religions, or countries impacts the model's predictions compared to the original training set.

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

The core objective of fairness testing is to evaluate a machine learning model's performance without bias, especially in contexts with implications for specific groups. This testing ensures that the model does not favor or discriminate against any group, aiming for unbiased results across all groups. Various tests, including those focused on attributes like gender, support this evaluation.

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

The goal of representation testing is to assess whether a dataset accurately represents a specific population or if biases within it could adversely affect the results of any analysis.

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

The primary goal of model robustness tests is to evaluate the model's ability to sustain consistent output when exposed to perturbations in the data it predicts. In tasks like Named Entity Recognition (NER), the focus is on assessing how variations in input data, such as documents with typos or fully uppercased sentences, impact the model's prediction performance compared to documents similar to those in the original training set.

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

The primary goal of toxicity tests is to assess the ideological toxicity score of a given text, specifically targeting demeaning speech based on political, philosophical, or social beliefs. This includes evaluating instances of hate speech rooted in individual ideologies, such as feminism, left-wing politics, or right-wing politics

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

The primary objective of the sensitivity test is to assess the model's responsiveness when introducing negation and toxic words, gauging its level of sensitivity in these scenarios.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Sensitivity](sensitivity)       | [Negation](sensitivity#negation)                                                                                | `question-answering`                                                                  |
| [Sensitivity](sensitivity)       | [Toxicity](sensitivity#toxicity)                                                                                | `question-answering`                                                                  |

## Sycophancy Tests

The primary goal of addressing sycophancy in language models is to mitigate undesirable behaviors where models tailor their responses to align with a human user's view, even when that view is not objectively correct.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Sycophancy](sycophancy)       | [Sycophancy Math](sycophancy#sycophancy_math)                                                                                | `question-answering`                                                                  |
| [Sycophancy](sycophancy)       | [Sycophancy NLP](sycophancy#sycophancy_nlp)                                                                                | `question-answering`                                                                  |

## Stereotype Tests

The primary goal of stereotype tests is to evaluate how well models perform when confronted with common gender stereotypes, occupational stereotypes, or other prevailing biases. In these assessments, models are scrutinized for their propensity to perpetuate or challenge stereotypical associations, shedding light on their capacity to navigate and counteract biases in their predictions.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Stereotype](stereotype)           | [gender-occupational-stereotype](stereotype#gender-occupational-stereotype)                                      | `fill-mask`                                                                         |
| [Stereotype](stereotype)       | [common-stereotypes](stereotype#common-stereotypes)                                                            | `fill-mask`                                                                       |

## StereoSet Tests

The primary goal of StereoSet is to provide a comprehensive dataset and method for assessing bias in Language Models (LLMs). Utilizing pairs of sentences, StereoSet contrasts one sentence that embodies a stereotypic perspective with another that presents an anti-stereotypic view. This approach facilitates a nuanced evaluation of LLMs, shedding light on their sensitivity to and reinforcement or mitigation of stereotypical biases.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [StereoSet](stereoset)           | [intersentence](stereoset#intersentence)                                                                        | `question-answering`                                                                         |
| [StereoSet](stereoset)           | [intrasentence](stereoset#intrasentence)                                                                        | `question-answering`                                                                         |

## Ideology Tests

The Idology Test is a tool assessing political beliefs on a two-dimensional grid, surpassing the traditional left-right spectrum. The Political Compass aims for a nuanced understanding, avoiding oversimplification and capturing the full range of political opinions and beliefs.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Ideology](ideology)           | [Political Compass](ideology#political_compass)                                                                        | `ideology`|

## Legal Tests

The primary goal of the Legal Test is to assess a model's capacity to reason about the strength of support provided by a given case summary. This evaluation aims to gauge the model's proficiency in legal reasoning and comprehension.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Legal](Legal)                   | [legal-support](legal#legal-support)                                                                            | `question-answering`                                                                       |

## Clinical Test

The Clinical Test evaluates the model for potential demographic bias in suggesting treatment plans for two patients with identical diagnoses. This assessment aims to uncover and address any disparities in the model's recommendations based on demographic factors.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Clinical](clinical)             | [demographic-bias](clinical#demographic-bias)                                                                   | `text-generation`                                                                    |

## Security Test

The Security Test, featuring the Prompt Injection Attack, is designed to assess prompt injection vulnerabilities in Language Models (LLMs). This test specifically evaluates the model's resilience against adversarial attacks, gauging its ability to handle sensitive information appropriately and ensuring robust security measures

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Security](security)             | [prompt_injection_attack](security#prompt_injection_attack)                                                     | `text-generation`                                                                          |

## Disinformation Test

The Disinformation Test aims to evaluate the model's capacity to generate disinformation. By presenting the model with disinformation prompts, the experiment assesses whether the model produces content that aligns with the given input, providing insights into its susceptibility to generating misleading or inaccurate information.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Disinformation](disinformation) | [Narrative Wedging](disinformation#narrative_wedging)                                                           | `text-generation`                                                               |


## Factuality Test

The Factuality Test is designed to evaluate the ability of language models (LLMs) to determine the factuality of statements within summaries. This test is particularly relevant for assessing the accuracy of LLM-generated summaries and understanding potential biases that might affect their judgments. 

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Factuality](factuality)         | [Order Bias](factuality#order_bias)                                                                             | `question-answering`                                                                   |                                                               


## Grammar Test

The Grammar Test assesses language models' proficiency in intelligently identifying and correcting intentional grammatical errors, ensuring refined language understanding and enhancing overall processing quality.

{:.table-model-big}
| Test Category                    | Test Name                                                                                                       | Supported Tasks                                                                      |
| -------------------------------- |
| [Grammar](grammar)         | [Paraphrase](grammar#paraphrase)                                                                             | `text-classification`, `question-answering`                                                                   |                                                               

</div></div>