# MACROSCORE Experimentation Repo

This repository contains all the code and experiments that has been conducted for the MACROSCORE project.

Project link: [https://usc-isi-i2.github.io/macro-score/](https://usc-isi-i2.github.io/macro-score)

For missing data & model files, download them from this google drive link: 
[https://drive.google.com/drive/folders/1pOQ-UakIQNZzcgaxXITQSr7MrGgbTMVq?usp=sharing](https://drive.google.com/drive/folders/1pOQ-UakIQNZzcgaxXITQSr7MrGgbTMVq?usp=sharing)

# 1. Folder Structure:
The folder structure for this repository is described below,

`\data_processing_utils` = Contains some code utilities for data exploration and data preprocessing. 
(These are written by the previous students. They are currently not in use and are only kept as legacy code.)

`\feature_extraction` = Contains code utilities to train and test the feature extraction task for MACROSCORE. Refer to the
[feature_extraction/README.md](feature_extraction/README.md) for further details.

`\lm_classify` = Contains code utilities to train and test the reproducibility classification or reproducibility
prediction task for MACROSCORE. Refer to the [lm_classify/README.md](lm_classify/README.md) for further details.

`\phrase_level_extraction` = Contains code utilities to extract important trigger phrases for both Text classification 
task and Feature Extraction task. Modified from [https://github.com/INK-USC/hierarchical-explanation-neural-sequence-models] (https://github.com/INK-USC/hierarchical-explanation-neural-sequence-models).
Refer to the [phrase_level_extraction/README.md](phrase_level_extraction/README.md) for further details.

`\score_utils` = Contains code utilities to perform certain specific experiments and tasks.

`\tagtog_annotations` = Contains code utilities to transform the manually annotated NER html pages downloaded from
TagTog and convert to BIO-encoded NER data. 



# 2. Folders available on Google drive:
`\data`  = Contains all the raw datasets that were given to us by DARPA or by manual tagging. 

`\data_processed` = Contains all the processed datasets that were used to conduct the experiments.


























