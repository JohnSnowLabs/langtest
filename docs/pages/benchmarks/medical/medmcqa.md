---
layout: docs
header: true
seotitle: MedMCQA Benchmark | LangTest | John Snow Labs
title: MedMCQA
key: benchmarks-medmcqa
permalink: /docs/pages/benchmarks/medical/medmcqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---



**Source:** [MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering](https://proceedings.mlr.press/v174/pal22a)

The MedMCQA is a large-scale benchmark dataset of Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. 


{:.table2}
| subsets       | Details                                                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MedMCQA-Test**    | This dataset does not contain labels, so accuracy and fairness tests cannot be run on it. Only robustness tests can be applied.                             |
| **MedMCQA-Validation** | This dataset does contain labels, enabling the execution of robustness, accuracy, and fairness tests. |


Both the subset contains the following splits:

- Anaesthesia
- Anatomy
- Biochemistry
- Dental
- ENT
- Forensic_Medicine
- Gynaecology_Obstetrics
- Medicine
- Microbiology
- Ophthalmology
- Pathology
- Pediatrics
- Pharmacology
- Physiology
- Psychiatry
- Radiology
- Skin
- Social_Preventive_Medicine
- Surgery
- Unknown

## Benchmarks

{:.table3}
| Model                               | uppercase | lowercase	 | titlecase | add_typo | dyslexia_word_swap | add_abbreviation | add_slangs | add_speech_to_text_typo | add_ocr_typo | adjective_synonym_swap	 |
|--------------------------------------|:---------:|:---------:|:---------:|:-------:|:-------------------:|:----------------:|:----------:|:------------------------:|:------------:|:-----------------------:|:------------------------:|
| j2-jumbo-instruct                   |    58%    |    67%    |    67%    |   81%   |         83%         |       81%        |     76%    |           80%            |      75%      |           79%           |
| j2-grande-instruct                  |    55%    |    62%    |    66%    |   80%   |         82%         |       81%        |     80%    |           80%            |      76%      |           81%           |
| gpt-3.5-turbo-instruct              |    81%    |    85%    |    81%    |   87%   |         87%         |      87%        |     79%    |          86%            |     83%      |           84%           |
| mistralai/Mistral-7B-Instruct-v0.1  |    57%    |    61%    |    62%    |   86%   |         85%         |       84%        |     79%    |           84%            |      78%      |           84%           |

**Dataset info:**
- subset: MedMCQA-Test
- Split: Medicine, Anatomy, Forensic_Medicine, Microbiology, Pathology, Anaesthesia, Pediatrics, Physiology, Biochemistry, Gynaecology_Obstetrics, Skin, Surgery, Radiology
- Records: 3433
- Testcases: 27639
- Evaluatuion: GPT-3.5-Turbo-Instruct

![MedMCQA Benchmark](/assets/images/benchmarks/medmcq.png)
