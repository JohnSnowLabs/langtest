---
layout: docs
header: true
seotitle: MedQA Benchmark | LangTest | John Snow Labs
title: MedQA
key: benchmarks-medqa
permalink: /docs/pages/benchmarks/medical/medqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---


**Source:** [What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams](https://paperswithcode.com/dataset/medqa-usmle)

The MedQA is a benchmark dataset of Multiple choice question answering based on the United States Medical License Exams (USMLE). The dataset is collected from the professional medical board exams.

You can see which subsets and splits are available below.

{:.table2}
| Split           |Details                                                                                                |
| ----------------|------------------------------------------------------------------------------------------------------ |
| **test**       | Testing set from the MedQA dataset, containing 1273 question and answers examples.                         |
| **test-tiny**  | Truncated version of the test set from the MedQA dataset, containing 50 question and answers examples. |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Benchmarks

{:.table3}
| Models                         | uppercase | lowercase | titlecase	 | add_typo | dyslexia_word_swap | add_abbreviation	 | add_slangs | add_speech_to_text_typo	 | add_ocr_typo | adjective_synonym_swap |
|:-------------------------------:|:---------:|:---------:|:---------:|:--------:|:-------------------:|:----------------:|:----------:|:-----------------------:|:------------:|:-----------------------:|
| j2-jumbo-instruct               |    47%    |    56%    |    54%    |    76%   |         74%         |        70%       |    73%     |           76%           |      66%      |           71%           |
| j2-grande-instruct              |    54%    |    64%    |    61%    |    86%   |         84%         |        79%       |    87%     |           85%           |      76%      |           82%           |
| gpt-3.5-turbo-instruct          |    80%    |    86%    |    85%    |    92%   |         91%         |        85%       |    90%     |           92%           |      85%      |           88%           |


> Minimum pas rate: 75%

**Dataset info:**
- Split: test
- Records: 1273
- Testcases: 12626
- Evaluatuion: GPT-3.5-Turbo-Instruct

</div>

![MedQA Benchmark](/assets/images/benchmarks/medqa.png)
