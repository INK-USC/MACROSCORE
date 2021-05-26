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

`\score_utils` = Contains code utilities to perform certain specific experiments and tasks. It has the following 
file structure,

```
score_utils
    ├── notebooks
            ├── Claim-match Experiements.ipynb, 
                Claim-match Experiements - Coordinates Based.ipynb                      (Experiments about the claim matching task using text and coordinates)
            ├── Cutpoint Identification.ipynb                                           (Cutpoint threshold identification of various features for the TA2 dataset)
            ├── Feature Extractions Stats.ipynb                                         (Statistics about the feature extraction results for TA2 dataset)
            ├── Important Phrase_Section Analysis for TA2 overall paper content.ipynb, 
                Post-hoc Analysis on phrase-level extractions.ipynb                     (Statistics/Analysis about the important phrase extraction results for reproducibility classification task)
            └── Phrase Extractions Analysis for NER task.ipynb                          (Statistics/Analysis about the important phrase extractions results for Feature extraction task)
    
    ├── createDataFolds.py,
        createSOCData.py                (Create the data folds for running the k-fold cross validation based model training, used for reproducibility classification task in SOC)
    ├── createNERTCdata.py              (Convert the BIO formated NER dataset into text classification based NER dataset which is used by SOC)
    ├── createTriggerNERData.py         (Create the TriggerNER format dataset based on automatic trigger extractions results)
    ├── getParseTrees.py,  
        getParseTreesNERTC.py           (Utilities related to creating parse trees for a input sentence, used for AutoTrigger)
    ├── matchClaims.py                  (Perform claim matching using the text for the entire TA2 dataset and compute the performance)
    ├── prepareBERTLMFinetuningData.py  (Create the domain-specific fine-tuning corpus from the reproduciblity classificaiton task dataset, used for LM training in SOC)
    └── tree_parsing_ref.py             (Utilities related to working with parse trees for a input sentence, used for AutoTrigger)
```

`\tagtog_annotations` = Contains code utilities to transform the manually annotated NER html pages downloaded from
TagTog and convert to BIO-encoded NER data. For more details about using this folder, refer to the [REAMDE_From_dongho.md](README_from_dongho.md).


# 2. Folders available on Google drive:
`\data`  = Contains all the raw datasets that were given to us by DARPA or by manual tagging. 

`\data_processed` = Contains all the processed datasets that were used to conduct the experiments.


























