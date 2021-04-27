# SCORE Micro Feature Extraction


## SciBERT-based Feature Extraction

To train the feature extraction model using SciBERT, use the following command,

```
python3 flair_train.py --data_dir ../../data_processed/ner/ner_dataset 
                       --output_dir flair_models/scibert_ner_model_finetuned_with_train_only 
                       --num_epochs 200
```

where, <br>
data_dir = Data directory that contains the NER data in BIO-encoded format. 
Must contain train.txt and test.txt files <br>
output_dir = Output directory to store the trained models <br>


