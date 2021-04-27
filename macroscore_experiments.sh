############ Feature Extraction Experiments ############

# 1. Train

# Train with train.txt and test.txt (done)
CUDA_VISIBLE_DEVICES=9 python3 flair_train.py --data_dir ../../data_processed/ner_dataset --output_dir flair_models/scibert_ner_model_finetuned_with_train_only --num_epochs 200

# Train with trainplustest.txt and test_new.txt (done)
CUDA_VISIBLE_DEVICES=3 python3 flair_train.py --data_dir ../../data_processed/ner_dataset_v2 --output_dir flair_models/scibert_ner_model --num_epochs 200

# Train with train.txt and test.txt with citations (done)
CUDA_VISIBLE_DEVICES=2 python3 flair_train.py --data_dir ../../data_processed/ner_dataset_with_citations --output_dir flair_models/scibert_ner_model_with_citations --num_epochs 60

# Train with train.txt and test.txt with glove embedding (done)
CUDA_VISIBLE_DEVICES=9 python3 flair_train.py --data_dir ../../data_processed/ner_dataset --output_dir flair_models/glove_ner_model_with_train_only --num_epochs 200


# 2. Predict/Infer
mkdir flair_pred_outputs
CUDA_VISIBLE_DEVICES=0 python3 flair_predict.py --checkpoint_dir flair_models_old --output_dir flair_pred_outputs

CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models/scibert_ner_model/final-model.pt --output_path flair_pred_output.json

CUDA_VISIBLE_DEVICES=0 python3 flair_predict.py

# 3. Evaluation


# Debug experiments
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models/scibert_ner_model/best-model.pt --output_path flair_pred_output.json


############ LM Experiments ############

############ Classification Experiments ############

# M1 with TA2 data:

# With fold 1 data (done)
CUDA_VISIBLE_DEVICES=5 python3 lm_train_with_CV.py --model_type M1 --loss_type classification --output_dir ta2_class_checkpoint/m1_folds --data_dir ../data_processed/ta2_classify_folds --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5

# Extract important segments (done)
CUDA_VISIBLE_DEVICES=1 python3 lm_extract_important_segments.py --model_type M1 --loss_type classification --data_dir ../data_processed/ta2_classify_folds --output_dir ../data_processed/ta2_data/ta2_classify_with_imp_segments


# M2 with TA2 data:

### Use sentence tokenized segments with max num tokens 100.
# With fold 1 data (done)
CUDA_VISIBLE_DEVICES=4 python3 lm_train_with_CV.py --model_type M2 --loss_type classification --output_dir ta2_class_checkpoint/m2_folds --data_dir ../data_processed/ta2_classify_folds --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5

# Extract important segments (done)
CUDA_VISIBLE_DEVICES=1 python3 lm_extract_important_segments.py --model_type M2 --loss_type classification --data_dir ../data_processed/ta2_classify_folds --output_dir ../data_processed/ta2_data/ta2_classify_with_imp_segments_sectionwise

### Use sentence as segments with max num tokens 100 (works best)
# With fold 1 data (done)
CUDA_VISIBLE_DEVICES=2 python3 lm_train_with_CV.py --model_type M2 --loss_type classification --output_dir ta2_class_checkpoint/m2_folds_sentwise --data_dir ../data_processed/ta2_classify_folds --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5
CUDA_VISIBLE_DEVICES=1 python3 lm_train_with_CV.py --model_type M2 --use_bilstm --loss_type classification --output_dir ta2_class_checkpoint/m2_folds_sentwise_bilstm --data_dir ../data_processed/ta2_classify_folds --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5

# Extract important segments (done)
CUDA_VISIBLE_DEVICES=2 python3 lm_extract_important_segments.py --model_type M2 --model_path ta2_class_checkpoint/m2_folds_sentwise/best_model.tar --loss_type classification --data_dir ../data_processed/ta2_classify_folds --output_dir ../data_processed/ta2_data/ta2_classify_with_imp_sentences
CUDA_VISIBLE_DEVICES=1 python3 lm_extract_important_segments.py --model_type M2 --use_bilstm --model_path ta2_class_checkpoint/m2_folds_sentwise_bilstm/best_model.tar --loss_type classification --data_dir ../data_processed/ta2_classify_folds --output_dir ../data_processed/ta2_data/ta2_classify_with_imp_sentences_bilstm


# Old experiments:
python3 RPP_scibert.py --feature_type content --data_type TA2 --data_path ../data_processed/ta2_data/TA2_scienceparse_classify_data_full_with_labels.json --output_dir scibert_ta2  --num_epochs 5 --num_folds_cv 5



# M3 with TA2 data:




############ Regression Experiments ############

# M1, M2 and M3-based Document Score Prediction: (Regression problem)

# M1 with TA2 data:

# With fold 1 data (done)
CUDA_VISIBLE_DEVICES=2 python3 lm_train.py --do_eval --model_type M1 --loss_type regression --output_dir ta2_reg_checkpoint/m1_fold1 --data_type TA2 --data_dir ../data_processed/ta2_folds/fold_1 --num_epochs 10

# 10 fold CV data (done)
CUDA_VISIBLE_DEVICES=2 python3 lm_train_with_folds_v2.py --model_type M1 --loss_type regression --output_dir ta2_reg_checkpoint/m1_folds --data_dir ../data_processed/ta2_folds/ta2_reg_folds/fold_full --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5,6,7,8,9,10

# Final model trained on whole data (ongoing)
CUDA_VISIBLE_DEVICES=2 python3 lm_train_with_folds_v2.py --model_type M1 --loss_type regression --output_dir ta2_reg_checkpoint/m1_final --data_dir ../data_processed/ta2_folds/fold_full --num_epochs 15 --train_batch_size 32 --eval_batch_size 32 --cv_folds -1


# M2 with TA2 data:

# With fold 1 data (done)
CUDA_VISIBLE_DEVICES=3 python3 lm_train.py --do_eval --model_type M2 --loss_type regression --output_dir ta2_reg_checkpoint/m2_fold1 --data_type TA2 --data_dir ../data_processed/ta2_folds/fold_1 --num_epochs 10

# 10 fold CV data (done)
CUDA_VISIBLE_DEVICES=2 python3 lm_train_with_folds_v2.py --model_type M2 --loss_type regression --output_dir ta2_reg_checkpoint/m2_folds --data_dir ../data_processed/ta2_folds/fold_full --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5,6,7,8,9,10

# Final model trained on whole data (ongoing)
CUDA_VISIBLE_DEVICES=2 python3 lm_train_with_folds_v2.py --model_type M2 --loss_type regression --output_dir ta2_reg_checkpoint/m2_final --data_dir ../data_processed/ta2_folds/fold_full --num_epochs 10 --train_batch_size 32 --eval_batch_size 32 --cv_folds -1


# M3 with TA2 data:

# With fold 1 data (done)
CUDA_VISIBLE_DEVICES=4 python3 lm_train.py --do_eval --model_type M3 --loss_type regression --output_dir ta2_reg_checkpoint/m3_fold1 --data_type TA2 --data_dir ../data_processed/ta2_folds/fold_1 --num_epochs 10

# 10 fold CV data (done)
CUDA_VISIBLE_DEVICES=0 python3 lm_train_with_folds_v2.py --model_type M3 --loss_type regression --output_dir ta2_reg_checkpoint/m3_folds --data_dir ../data_processed/ta2_folds/fold_full --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1,2,3,4,5,6,7,8,9,10

# Final model trained on whole data (ongoing)
CUDA_VISIBLE_DEVICES=3 python3 lm_train_with_folds_v2.py --model_type M3 --loss_type regression --output_dir ta2_reg_checkpoint/m3_final --data_dir ../data_processed/ta2_folds/fold_full --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds -1


# Debug experiments:
CUDA_VISIBLE_DEVICES=2 python3 lm_train.py --do_eval --model_type M1 --loss_type regression --output_dir temp --data_dir ../data_processed/ta2_folds/sample_fold --num_epochs 5
CUDA_VISIBLE_DEVICES=2 python3 lm_train.py --do_eval --model_type M2 --loss_type regression --output_dir temp --data_dir ../data_processed/ta2_folds/sample_fold --num_epochs 5
CUDA_VISIBLE_DEVICES=3 python3 lm_train.py --do_eval --model_type M3 --loss_type regression --output_dir temp2 --data_dir ../data_processed/ta2_folds/sample_fold --num_epochs 5

# Debug with mini-batches
CUDA_VISIBLE_DEVICES=0 python3 lm_train_v2.py --do_eval --model_type M1 --loss_type regression --output_dir temp --data_dir ../data_processed/ta2_folds/sample_fold --num_epochs 5
CUDA_VISIBLE_DEVICES=0 python3 lm_train_v2.py --do_eval --model_type M2 --loss_type regression --output_dir ta2_reg_checkpoint/m2_fold1_v2 --data_type TA2 --data_dir ../data_processed/ta2_folds/fold_1 --num_epochs 20 --train_batch_size 32 --eval_batch_size 32

# Debug with CV mini-batches
CUDA_VISIBLE_DEVICES=1 python3 lm_train_with_folds_v2.py --model_type M2 --loss_type regression --output_dir temp22 --data_dir ../data_processed/ta2_folds/fold_full --num_epochs 3 --train_batch_size 3 --eval_batch_size 3 --cv_folds 1,2


############ Other Experiments ############

# Debug NER models:
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models_v2/flair_models/scibertnew_1/final-model.pt --output_path pred/flair_pred_output1.json
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models_v2/flair_models/scibertnew_3/final-model.pt --output_path pred/flair_pred_output3.json
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models_v2/flair_models/scibertnew_4/final-model.pt --output_path pred/flair_pred_output4.json
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models_v2/flair_models/scibertnew_0/final-model.pt --output_path pred/flair_pred_output0.json
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models/scibert_ner_model/final-model.pt --output_path pred/flair_pred_output_ta1_5.json
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models_v2/flair_models/scibert_0/best-model.pt --output_path pred/flair_pred_output00.json
CUDA_VISIBLE_DEVICES=5 python3 flair_predict_v2.py --checkpoint_path flair_models_v2/flair_models/scibertnew_1/final-model.pt --output_path pred/flair_pred_output_ta1_5.json

# Debug M1 with folds:
CUDA_VISIBLE_DEVICES=1 python3 lm_train_with_folds.py --model_type M1 --loss_type regression --output_dir temp --data_type TA2 --data_dir ../data_processed/ta2_folds/sample_fold_v2 --num_epochs 5 --cv_folds -1
CUDA_VISIBLE_DEVICES=1 python3 lm_train_with_folds.py --model_type M1 --loss_type classification --output_dir temp --data_type TA2 --data_dir ../data_processed/ta2_folds/sample_fold_v2 --num_epochs 5 --train_batch_size 4 --eval_bath_size 4 --cv_folds 2




CUDA_VISIBLE_DEVICES=1 python3 PhraseExtractionModule.py

python3 lm_train_with_CV.py --model_type M2 --loss_type classification --output_dir temp --data_dir ../data_processed/ta2_classify_folds/fold_full --num_epochs 20 --train_batch_size 32 --eval_batch_size 32 --cv_folds 1
