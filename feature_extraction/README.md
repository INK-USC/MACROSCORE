# SCORE Micro Feature Extraction


## 1. SciBERT-based Feature Extraction

### 1.1 Training
To train the feature extraction NER model using SciBERT, use the following command,

```
python3 flair_train.py --data_dir ../../data_processed/ner/ner_dataset_cleaned \
                       --output_dir flair_models/scibert_ner_model \
                       --num_epochs 200 \
                       --gpu_device cuda:0
```

where, <br>
data_dir = Data directory that contains the NER data in BIO-encoded format.
Must contain train.txt and test.txt files <br>
output_dir = Output directory to store the trained models <br>
gpu_device = GPU Device number <br>

### 1.2 Inference
To extract the features on the new dataset using the already trained SciBERT NER model, 
use the following command,

```
python3 flair_predict.py --checkpoint_path flair_models/scibert_ner_model/best-model.pt \
                         --data_type TA1 \
                         --raw_data_path ../../data_processed/others/raw_all.txt \
                         --output_path prediction_out.json \
                         --gpu_device cuda:0
```
where, <br>
checkpoint_path = Model path to load the trained SciBERT NER model. <br>
data_type = Data type of the dataset. Valid Options: RPP, TA1. <br>
raw_data_path = Raw data path of the dataset. Should contain the parsed PDFs in .txt format. <br>
output_path = Output path to store the prediction outputs <br>
gpu_device = GPU Device number <br>

Before running the inference on the new dataset, use the previous command to train the 
SciBERT NER model.

**Note:** The flair_predict.py doesn't have support to run inference on the TA2 dataset. 
So, an TODO is to add this functionality to this script. 
For running inference on TA2 dataset for feature extraction task, use the production codebase. 
