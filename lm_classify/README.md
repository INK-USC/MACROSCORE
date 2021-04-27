# SCORE Reproducibility Classification/Prediction
 

### Training a classification Model 
 
To train various models for the reproducibility classification task 
along with k-fold cross validation, use the following command,

```
python3 lm_train_with_CV.py --model_type M2
                            --loss_type classification 
                            --output_dir trained_models/ta2_class_checkpoint/m2_folds
                            --data_dir ../data_processed/ta2_folds/ta2_classify_folds/fold_full 
                            --num_epochs 20 
                            --train_batch_size 32 
                            --eval_batch_size 32 
                            --cv_folds 1,2,3,4,5
```

where, <br>
model_type = Model type. Valid options: M1 or M2 or M3. <br>
M1 = SciBERT encoder + using only the claims from the paper <br>
M2 = SciBERT encoder + using the entire paper content <br>
M3 = Longformer encoder + using the entire paper content <br>

output_dir = Output directory to store the trained models <br>
data_dir = Data directory which contains the data file <br>
cv_folds = List of Fold ids to evaluate the model. 
Data file should have a column named "Fold_Id". 

### Extract Important Segments in a paper using a classification Model 

To extract important segments of the paper using the trained classification model, 
use the following command,

```
python3 lm_extract_important_segments.py --model_type M2 
                                         --model_path trained_models/ta2_class_checkpoint/m2_folds_sentwise_bilstm/fold_1/best_model.tar
                                         --loss_type classification 
                                         --data_dir ../data_processed/ta2_folds/ta2_classify_folds/fold_full 
                                         --output_dir ../data_processed/ta2_data/ta2_classify_with_imp_segments_sectionwise
```

where, <br>
model_type = Model type. Valid options: M1 or M2 or M3. <br>
M1 = SciBERT encoder + using only the claims from the paper <br>
M2 = SciBERT encoder + using the entire paper content <br>
M3 = Longformer encoder + using the entire paper content <br>

model_path = Model path of the trained model. <br>
output_dir = Output directory to store the extracted segments for all papers <br>
data_dir = Data directory which contains the data file <br>


### Training a Regression Model 

Similarly, to train various models for the reproducibility score prediction task 
along with k-fold cross validation, use the following command,

```
python3 lm_train_with_folds_v2.py --model_type M2
                                  --loss_type regression 
                                  --output_dir trained_models/ta2_reg_checkpoint/m2_folds 
                                  --data_dir ../data_processed/ta2_folds/ta2_reg_folds/fold_full 
                                  --num_epochs 20 
                                  --train_batch_size 32 
                                  --eval_batch_size 32 
                                  --cv_folds 1,2,3,4,5,6,7,8,9,10
```

