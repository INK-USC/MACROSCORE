### Directories (sorted by names)
```
├── CORD-19-research-challenge (Newly added data)
├── data
│   └── SCORE_json.json (Metadata for the TA1 papers)
├── data_processed
│   ├── conll_format.py (input: train.txt, test.txt, output: eng.train.src ...)
│   ├── raw_all*.txt    (un-annotated data for predicting; raw_all: TA1 papers)
│   ├── train.txt, test.txt    (annotated TA1 data for training & testing)
│   └── data.json       (intermedia file)
├── flair_models        (saved flair models & training logs)
├── flair_pred          (predictions)
├── glove               (glove embeddings)
├── logs                (model evaluation logs)
├── pdfs                (TA1 papers and RPP papers in pdf)
├── tagtog
│   ├── annotated       (annotated TA1 papers)
│   ├── plain.html*     (papers parsed from PDF using tagtog)
│   ├── text*           (paper text extracted from plain.html*)
│   └── tokenized*      (tokenized paper text from text*)
└── xml                 (TA1 papers in xml)
```

### Scripts (sorted by order of execution)
```
├── read_tagtog.py      (process tagtog/plain.html* into tagtog/text* and tagtog/tokenized*)
├── process_annotations.py  (process tagtog/annotated into data_processed/data.json)
├── inspect_annotations.py  (have a glance at the annotations)
├── make_data.py        (process data_processed/data.json into data_processed/train.txt and test.txt)
├── flair*              (scripts for training & evaluating & predicting of the flair models
                         train: data_processed/train.txt, test: test.txt, predict: raw_all*.txt)
├── process_raw.py      (process tagtog/tokenized* into data_processed/raw_all*.txt)
└── extract.py, phrases.py  (old script for automated phrase extraction)
```

### Procedures for building a NER model
1. Obtain papers in pdf (if papers already parsed, jump to 3.)
2. Parse pdfs using tagtog (into ```tagtog/plain.html*```)
3. Run ```read_tagtog.py``` to get tokenized text from parsed pdfs
4. Run ```process_annotations.py``` to get ```data_processed/data.json```
5. Run ```make_data.py``` to get training & testing data from ```data_processed/data.json``` to ```data_processed/train.txt, test.txt```
6. Run ```flair_train.py``` (in ```flair_run.sh```) to train a model
7. Run ```flair_eval.py``` (in ```flair_run.sh```) to test
8. Run ```process_raw.py``` to get ```data_processed/raw_all*.txt``` for predicting
9. Run ```flair_predict.py``` to make prediction
10. Final result is in ```flair_pred/```
