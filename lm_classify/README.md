# SCORE Reproducibility Classification/Prediction
 

## 1. Training a classification Model 
 
To train various models for the reproducibility classification task 
along with k-fold cross validation, use the following command,

```
python3 lm_train_with_CV.py --model_type M2 \
                            --loss_type classification \
                            --output_dir trained_models/ta2_class_checkpoint/m2_folds \
                            --data_dir ../data_processed/ta2_folds/ta2_classify_folds/fold_full \ 
                            --num_epochs 20 \
                            --train_batch_size 32 \
                            --eval_batch_size 32 \
                            --cv_folds 1,2,3,4,5
```

where, <br>
model_type = Model type. Valid options: M1 or M2 or M3. <br>
M1 = SciBERT encoder + using only the claims from the paper <br>
M2 = SciBERT encoder + using the entire paper content <br>
M3 = Longformer encoder + using the entire paper content <br>

output_dir = Output directory to store the trained models <br>
data_dir = Data directory which contains the data file <br>
num_epochs = Number of epochs to train the model <br>
train_batch_size = Batch size for training <br>
eval_batch_size = Batch size for evaluation or testing <br>
cv_folds = List of Fold ids to evaluate the model. 
Data file should have a column named "Fold_Id". 

To train using the entire data for a fixed number of epochs without using the k-fold cross validation, use the following command,

```
python3 lm_train_with_CV.py --model_type M2 \
                            --loss_type classification \
                            --output_dir trained_models/ta2_class_checkpoint/m2_final \
                            --data_dir ../data_processed/ta2_folds/ta2_classify_folds/fold_full \
                            --num_epochs 10 \
                            --train_batch_size 32 \
                            --eval_batch_size 32 \
                            --cv_folds -1
```

### 1.1 Extract Important Segments in a paper using a classification Model 

To extract important segments of the paper using the already trained classification model, 
use the following command,

```
python3 lm_extract_important_segments.py --model_type M2 \
                                         --model_path trained_models/ta2_class_checkpoint/m2_folds_sentwise_bilstm/best_model.tar \
                                         --loss_type classification \
                                         --data_dir ../data_processed/ta2_folds/ta2_classify_folds/fold_full \
                                         --output_dir ../results/ta2_classify_with_imp_segments_sectionwise
```

where, <br>
model_type = Model type. Valid options: M1 or M2 or M3. <br>
M1 = SciBERT encoder + using only the claims from the paper <br>
M2 = SciBERT encoder + using the entire paper content <br>
M3 = Longformer encoder + using the entire paper content <br>

model_path = Model path of the trained model. <br>
data_dir = Data directory which contains the data file <br>
output_dir = Output directory to store the extracted segments for all papers <br>

Before running this important segment extraction on test datasets, you can either run the training using the previous command (or) 
download the pretrained models from the google drive link: [https://drive.google.com/drive/folders/1EVcHi5k2ttynyQA9SCO1VmyYRS9b7Pvo?usp=sharing](https://drive.google.com/drive/folders/1EVcHi5k2ttynyQA9SCO1VmyYRS9b7Pvo?usp=sharing)
and put them in the path `MACROSCORE/lm_classify/`.


## 2. Training a Regression Model 

Similarly, to train various models for the reproducibility score prediction task 
along with k-fold cross validation, use the following command,

```
python3 lm_train_with_CV.py --model_type M2 \
                            --loss_type regression \
                            --output_dir trained_models/ta2_reg_checkpoint/m2_folds \
                            --data_dir ../data_processed/ta2_folds/ta2_reg_folds/fold_full \
                            --num_epochs 20 \
                            --train_batch_size 32 \
                            --eval_batch_size 32 \
                            --cv_folds 1,2,3,4,5,6,7,8,9,10
```

where the options are similar to that of the previous train command for classification task
except the `loss_type` argument which is not changed to `regression`.

To train using the entire data for a fixed number of epochs without using the k-fold cross validation, use the following command,

```
python3 lm_train_with_CV.py --model_type M2 \
                            --loss_type regression \
                            --output_dir trained_models/ta2_reg_checkpoint/m2_final \
                            --data_dir ../data_processed/ta2_folds/ta2_reg_folds/fold_full \
                            --num_epochs 10 \
                            --train_batch_size 32 \
                            --eval_batch_size 32 \
                            --cv_folds -1
```

