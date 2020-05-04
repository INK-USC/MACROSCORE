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
4. Manually add annotations (put into ```tagtog/annotated```)
5. Run ```process_annotations.py``` to get ```data_processed/data.json```
6. Run ```make_data.py``` to get training & testing data from ```data_processed/data.json``` to ```data_processed/train.txt, test.txt```
7. Run ```flair_train.py``` (in ```flair_run.sh```) to train a model
8. Run ```flair_eval.py``` (in ```flair_run.sh```) to test
9. Run ```process_raw.py``` to get ```data_processed/raw_all*.txt``` for predicting
10. Run ```flair_predict.py``` to make prediction
11. Final result is in ```flair_pred/```

Additional notes for ```flair_predict.py```: In this script I first use flair to predict on sentences, then use rule-based method to normalize the prediction (e.g., ```"R2 = 0.4"``` -> ```["R2", 0.4]```; ```"eighty-six"``` -> ```86```). I manually inspect on predictions and summarize patterns of extracted spans (```flair_pred/patterns.txt``` containing good and bad patterns), and then develop methods to process the patterns.

#### Update 04/13
New request: For TA1, find effect sizes (ES) and p-values (PV) around claim 4 (inferencial test, included in TA1 metadata). I instead calculate its distance (# sentences away) from the nearest claim 4 for every effect size and p-value extracted. Details can be found in the Explanation doc in the Google Drive.

#### Update 04/20
Added ```covid_analyze.py``` for extracting covid-19 papers (vs. papers talking about other coronavirus)
Todo:
1. Ask Jay for the claim annotations for CORD-19 papers
2. Look into Topic Forest: provides tree-like topics for CORE-19 papers to extract covid-19 related papers (http://topicforest.com/queryterm/covid_19/topictree)
3. Variable identification on a small subset of CORD: can discuss with Daniel

#### Update 05/03
Added ```find_SBS_papers_cord19.py``` to find CORD19 papers that are similar to TA1 papers (potential SBS papers)
Method:
1. Process TA1 papers for phrase extraction: data_processed/raw_all_autophrase.txt
2. Use AutoPhrase to extract key phrases from TA1 papers (I only used multi-word phrases)
3. Compute tf-idf matrices of key phrases for both TA1 and CORD19 papers, then compute cosine similarity: find_SBS_papers_cord19.py
4. Reorder the CORD19 papers based on cosine similarity: data/CORD19_reordered_with_similarity.csv
