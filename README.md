# MACROSCORE Experimentation Repo

This repository contains all the experiments that has been conducted for the MACROSCORE project.

Project link: https://usc-isi-i2.github.io/macro-score/

The folder structure is described below,

`\data`  = Contains all the raw datasets that were given to us by DARPA or by manual tagging. 

`\data_processed` = Contains all the processed datasets that were used to conduct the experiments.

`\data_processing_utils` = Contains some code utilities for data exploration and data preprocessing. 
(Written by the previous students)

`\feature_extraction` = Contains code utilities to train and test the feature extraction task for MACROSCORE

`\lm_classify` = Contains code utilities to train and test the reproducibility classification or reproducibility
prediction task for MACROSCORE.

`\phrase_level_extraction` = Contains code utilities to extract important trigger phrases for both Text classification 
task and Feature Extraction task. Modified from https://github.com/INK-USC/hierarchical-explanation-neural-sequence-models

`\score_utils` = Contains code utilities to perform certain specific tasks required by Danny/Fred or Jay. 

`\tagtog_annotations` = Contains code utilities to transform the manually annotated NER html pages downloaded from
TagTog and convert to BIO-encoded NER data. 

























