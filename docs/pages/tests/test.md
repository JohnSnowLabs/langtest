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
modify_date: "2024-01-31"
---

<div class="filter-section">
    <label for="task-selector">Select a Task:</label>
    <select id="task-selector">
        <option value="all">All Tasks</option>
        <option value="ner">NER</option>
        <option value="text-classification">Text Classification</option>
        <option value="question-answering">Question Answering</option>
        <option value="summarization">Summarization</option>
        <option value="text-generation">Text-Generation</option>
        <option value="fill-mask">Fill-Mask</option>
    </select>
</div>

<div id="combined-table-container"></div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        const taskSelector = document.getElementById('task-selector');
        const mainContent = document.querySelector('.main-docs');

        // Retrieve the saved filter and timestamp from localStorage
        const savedFilterData = localStorage.getItem('selectedTaskFilterData');
        const savedFilterDataObj = savedFilterData ? JSON.parse(savedFilterData) : null;
        const currentTime = new Date().getTime();

        if (savedFilterDataObj && (currentTime - savedFilterDataObj.timestamp) <= 30000) {
            // Apply the saved filter if it's within the 30-second timeframe
            taskSelector.value = savedFilterDataObj.filter;
            filterTests(savedFilterDataObj.filter);
        } else {
            // Default to 'All Tasks' and clear any saved filter if expired or not set
            taskSelector.value = 'all';
            document.getElementById('combined-table-container').innerHTML = '';
            mainContent.style.display = 'block';
            localStorage.removeItem('selectedTaskFilterData');
        }

        taskSelector.addEventListener('change', function() {
            // Save the selected filter along with the current timestamp
            const filterData = JSON.stringify({
                filter: this.value,
                timestamp: new Date().getTime()
            });
            localStorage.setItem('selectedTaskFilterData', filterData);

            if (this.value !== 'all') {
                filterTests(this.value);
            } else {
                // Reset to the default 'All Tasks' view without any filtering
                document.getElementById('combined-table-container').innerHTML = '';
                mainContent.style.display = 'block';
            }
        });

        function filterTests(task) {
            document.getElementById('combined-table-container').innerHTML = ''; 
            mainContent.style.display = 'none'; 

            var combinedTableHtml = '<table class="table-model-big"><thead><tr><th>Test Name</th><th>Description</th><th>Test Category</th></tr></thead><tbody>';
            var allRows = document.querySelectorAll('.table-model-big tr');

            allRows.forEach(function(row) {
                if (row.cells[1] && row.cells[1].innerHTML.includes('/docs/pages/task/' + task)) {
                    let category = 'General'; 
                    const link = row.querySelector('a');
                    if (link && link.getAttribute('href')) {
                        const parts = link.getAttribute('href').split('/');
                        const lastPart = parts[parts.length - 1];
                        category = lastPart.split('#')[0];
                        category = `<a href="/docs/pages/tests/${category}">${category}</a>`; 
                    }
                    combinedTableHtml += `<tr>${row.innerHTML}<td>${category}</td></tr>`;
                }
            });

            combinedTableHtml += '</tbody></table>';
            document.getElementById('combined-table-container').innerHTML = combinedTableHtml;
        }
    });
</script>

<style>
    .filter-section {
        margin-bottom: 20px;
    }
</style>





<div class="main-docs" markdown="1">
<div class="h3-box" markdown="1">

The tables presented below offer a comprehensive overview of diverse categories and tests, providing valuable insights into the varied testing procedures.

## Accuracy Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
| [Min F1 Score](accuracy#min-f1-score)      | [ner](/docs/pages/task/ner)  , [text-classification](/docs/pages/task/text-classification)                   |
| [Min Macro-F1 Score](accuracy#min-macro-f1-score) | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)               |
| [Min Micro-F1 Score](accuracy#min-micro-f1-score) | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)               |
| [Min Precision Score](accuracy#min-precision-score) | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)             |
| [Min Recall Score](accuracy#min-recall-score) | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)                 |
| [Min Weighted-F1 Score](accuracy#min-weighted-f1-score) | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)           |
| [Min Exact Match Score](accuracy#min-exact-match-score) | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)   |
| [Min BLEU Score](accuracy#min-bleu-score)  | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)           |
| [Min Rouge1 Score](accuracy#min-rouge1-score) | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)       |
| [Min Rouge2 Score](accuracy#min-rouge2-score) | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)       |
| [Min RougeL Score](accuracy#min-rougel-score) | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)       |
| [Min RougeLsum Score](accuracy#min-rougelsum-score) | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)   |
| [LLM Eval](accuracy#llm-eval) | [question-answering](/docs/pages/task/question-answering) |


## Bias Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                          | [Replace To Asian First Names](bias#replace-to-asian-firstnames)                                                | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Asian Last Names](bias#replace-to-asian-lastnames)                                                  | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Black First Names](bias#replace-to-black-firstnames)                                                | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Black Last Names](bias#replace-to-black-lastnames)                                                  | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Buddhist Names](bias#replace-to-buddhist-names)                                                     | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Christian Names](bias#replace-to-christian-names)                                                   | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Female Pronouns](bias#replace-to-female-pronouns)                                                   | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To High Income Country](bias#replace-to-high-income-country)                                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Hindu Names](bias#replace-to-hindu-names)                                                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Hispanic First Names](bias#replace-to-hispanic-firstnames)                                          | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Hispanic Last Names](bias#replace-to-hispanic-lastnames)                                            | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Interracial Last Names](bias#replace-to-inter-racial-lastnames)                                     | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Jain Names](bias#replace-to-jain-names)                                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Lower Middle Income Country](bias#replace-to-lower-middle-income-country)                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Low Income Country](bias#replace-to-low-income-country)                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Male Pronouns](bias#replace-to-male-pronouns)                                                       | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Muslim Names](bias#replace-to-muslim-names)                                                         | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Native American Last Names](bias#replace-to-native-american-lastnames)                              | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Neutral Pronouns](bias#replace-to-neutral-pronouns)                                                 | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Parsi Names](bias#replace-to-parsi-names)                                                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Sikh Names](bias#replace-to-sikh-names)                                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To Upper Middle Income Country](bias#replace-to-upper-middle-income-country)                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To White First Names](bias#replace-to-white-firstnames)                                                | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                          | [Replace To White Last Names](bias#replace-to-white-lastnames)                                                  | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |

## Fairness Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                        | [Max Gender F1 Score](fairness#max-gender-f1-score)                                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)                                                      |
                        | [Min Gender F1 Score](fairness#min-gender-f1-score)                                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)                                                      |
                        | [Min Gender Rouge1 Score](fairness#min-gender-rouge1-score)                                                     | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Min Gender Rouge2 Score](fairness#min-gender-rouge2-score)                                                     | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Min Gender RougeL Score](fairness#min-gender-rougeL-score)                                                     | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Min Gender RougeLSum Score](fairness#min-gender-rougeLsum-score)                                               | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Min Gender LLM Eval](fairness#min-gender-llm-eval)                                               | [question-answering](/docs/pages/task/question-answering)                                                                          |
                        | [Max Gender Rouge1 Score](fairness#max-gender-rouge1-score)                                                     | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Max Gender Rouge2 Score](fairness#max-gender-rouge2-score)                                                     | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Max Gender RougeL Score](fairness#max-gender-rougeL-score)                                                     | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Max Gender RougeLSum Score](fairness#max-gender-rougeLsum-score)                                               | [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                                              |
                        | [Max Gender LLM Eval](fairness#max-gender-llm-eval)                                               | [question-answering](/docs/pages/task/question-answering)                                                                           |

## Representation Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                | [Min Country Economic Representation Count](representation#min-country-economic-representation-count)           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Country Economic Representation Proportion](representation#min-country-economic-representation-proportion) | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Ethnicity Representation Count](representation#min-ethnicity-representation-count)                         | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Ethnicity Representation Proportion](representation#min-ethnicity-representation-proportion)               | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Gender Representation Count](representation#min-gender-representation-count)                               | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Gender Representation Proportion](representation#min-gender-representation-proportion)                     | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Label Representation Count](representation#min-label-representation-count)                                 | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)                                                      |
                | [Min Label Representation Proportion](representation#min-label-representation-proportion)                       | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification)                                                      |
                | [Min Religion Name Representation Count](representation#min-religion-name-representation-count)                 | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                | [Min Religion Name Representation Proportion](representation#min-religion-name-representation-proportion)       | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |

## Robustness Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                   | [Add Context](robustness#add-context)                                                                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)  , [translation](/docs/pages/task/translation)  |
                   | [Add Contraction](robustness#add-contraction)                                                                   | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Add Punctuation](robustness#add-punctuation)                                                                   | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)  , [translation](/docs/pages/task/translation)  |
                   | [Add Typo](robustness#add-typo)                                                                                 | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [American to British](robustness#american-to-british)                                                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [British to American](robustness#british-to-american)                                                           | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Lowercase](robustness#lowercase)                                                                               | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Strip Punctuation](robustness#strip-punctuation)                                                               | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Swap Entities](robustness#swap-entities)                                                                       | [ner](/docs/pages/task/ner)                                                                               |
                   | [Titlecase](robustness#titlecase)                                                                               | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Uppercase](robustness#uppercase)                                                                               | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)  , [translation](/docs/pages/task/translation)  |
                   | [Number to Word](robustness#number-to-word)                                                                     | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Add OCR Typo](robustness#add-ocr-typo)                                                                         | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) . [translation](/docs/pages/task/translation)   |
                   | [Dyslexia Word Swap](robustness#dyslexia-word-swap)                                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Add Slangs](robustness#add-slangs)                                                                             | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Add Speech to Text Typo](robustness#add-speech-to-text-typo)                                                   | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Add Abbreviations](robustness#add-abbreviation)                                                                | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization)                |
                   | [Multiple Perturbations](robustness#multiple-perturbations)                                                     | [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)          |
                   | [Adjective Synonym Swap](robustness#adjective-synonym-swap)                                                     | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Adjective Antonym Swap](robustness#adjective-antonym-swap)                                                     | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Strip All Punctution](robustness#strip-all-punctuation)                                                        | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |
                   | [Randomize Age](robustness#random-age)                                                                          | [ner](/docs/pages/task/ner), [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering) , [summarization](/docs/pages/task/summarization) , [translation](/docs/pages/task/translation)   |

## Toxicity Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                         | [general_toxicity](toxicity#general_toxicity)                                                                                 | [text-generation](/docs/pages/task/text-generation)                                                                           |
                         | [identity_attack](toxicity#identity_attack)                                                                                   | [text-generation](/docs/pages/task/text-generation)                                                                           |
                         | [insult](toxicity#insult)                                                                             | [text-generation](/docs/pages/task/text-generation)                                                                           |
                         | [homosexual_gay_or_lesbian](toxicity#lgbtqphobia)                                                                                       | [text-generation](/docs/pages/task/text-generation)                                                                           |
                         | [threat](toxicity#threat)                                                                                       | [text-generation](/docs/pages/task/text-generation)                                                                           |
                         | [obscene](toxicity#obscene)                                                                               | [text-generation](/docs/pages/task/text-generation)                                                                           |

## Sensitivity Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                     | [Add Negation](sensitivity#add-negation)                                                                                | [question-answering](/docs/pages/task/question-answering)                                                                   |
                     | [Add Toxic Words](sensitivity#add-toxic-words)                                                                                | [question-answering](/docs/pages/task/question-answering)                                                                   |

## Sycophancy Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                  | [Sycophancy Math](sycophancy#sycophancy_math)                                                                                | [question-answering](/docs/pages/task/question-answering)                                                                   |
                  | [Sycophancy NLP](sycophancy#sycophancy_nlp)                                                                                | [question-answering](/docs/pages/task/question-answering)                                                                   |

## Stereotype Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                         | [wino Bias](stereotype#wino-bias-huggingface)                                      | [fill-mask](/docs/pages/task/fill-mask) , [question-answering](/docs/pages/task/question-answering)                                                                            |
                     | [CrowS Pairs](stereotype#crows-pairs)                                                            | [fill-mask](/docs/pages/task/fill-mask)                                                                          |

## StereoSet Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
               | [intersentence](stereoset#intersentence)                                                                        | [question-answering](/docs/pages/task/question-answering)                                                                          |
               | [intrasentence](stereoset#intrasentence)                                                                        | [question-answering](/docs/pages/task/question-answering)                                                                          |

## Ideology Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                | [Political Compass](ideology#political_compass)                                                                        | [question-answering](/docs/pages/task/question-answering) |

## Legal Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                | [legal-support](legal#legal-support)                                                                            | [question-answering](/docs/pages/task/question-answering)                                                                        |

## Clinical Tests


{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                | [demographic-bias](clinical#demographic-bias)                                                                   | [text-generation](/docs/pages/task/text-generation)                                                                     |

## Security Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                | [prompt_injection_attack](security#prompt_injection_attack)                                                     | [text-generation](/docs/pages/task/text-generation)                                                                           |

## Disinformation Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                | [Narrative Wedging](disinformation#narrative_wedging)                                                           | [text-generation](/docs/pages/task/text-generation)                                                                |


## Factuality Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                    | [Order Bias](factuality#order_bias)                                                                             | [question-answering](/docs/pages/task/question-answering)                                                                    |                                                               


## Grammar Tests

{:.table-model-big}
| Test Name                                 | Supported Tasks                                  |
|-------------------------------------------|--------------------------------------------------|
                   | [Paraphrase](grammar#paraphrase)                                                                             | [text-classification](/docs/pages/task/text-classification), [question-answering](/docs/pages/task/question-answering)                                                                    |                                                               

</div></div>