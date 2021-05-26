# SCORE Phrase Level Extraction


## 1. For Reproducibility Classification Task
For performing the important phrase extraction for the reproducibility classification task,
follow the below steps,

### 1.1 Train the Reproducibility Classification Model (used in SOC)
First, train the reproducibility classification model using only the important claim
of the paper as input. This model predicts the reproducibility class label such as 0 or 1 as
the output. Train the model using the below command,

```
mkdir models
export model_path=repr_claims_bert
CUDA_VISIBLE_DEVICES=<gpu_num> python -m bert.run_classifier  --task_name repr \
                                                              --do_train \
                                                              --do_eval \
                                                              --do_lower_case \
                                                              --data_dir data/repr_claims \
                                                              --bert_model allenai/scibert_scivocab_uncased \
                                                              --max_seq_length 500 \
                                                              --train_batch_size 8 \
                                                              --learning_rate 2e-5 \
                                                              --num_train_epochs 20 \
                                                              --output_dir models/${model_path}
```

### 1.2 Train the Language Model (used for Context Sampling in SOC)
To sample the context during the Sampling and Occlusion (SOC) algorithm, train the 
LSTM based Language model on the training corpus using the below command,

```
export lm_path=repr_claims_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=<gpu_num> python -m lm.lm_train --task repr \
                                                     --save_path models/${lm_path} \
                                                     --lr 0.0002 \
                                                     --use_bert_tokenizer \
                                                     --batch_size 32 \
                                                     --epochs 40
```

### 1.3 Get Explanations using Sampling and Occlusion (SOC) Algorithm
To obtain the explanations using the Sampling and Occlusion (SOC) algorithm, use the 
below command,

```
export algo=soc
export exp_name=repr_claims_train_full
export model_path=repr_claims_bert
export lm_path=repr_claims_lstm_lm_bert_tokenized
mkdir repr
mkdir repr/results
CUDA_VISIBLE_DEVICES=<gpu_num> python explain.py    --resume_snapshot models/${model_path} \ 
                                                    --method ${algo} \
                                                    --lm_path models/${lm_path} \
                                                    --batch_size 1 \
                                                    --start 0 \
                                                    --stop 703 \
                                                    --exp_name ${exp_name} \
                                                    --task repr \
                                                    --explain_model bert \
                                                    --bert_model allenai/scibert_scivocab_uncased \
                                                    --use_bert_tokenizer \
                                                    --nb_range 10 \
                                                    --sample_n 20 \
                                                    --dataset train
```

**Note:** Run the above commands inside the [hiexpl_soc_only/](hiexpl_soc_only/) folder.

For a complete list of experiments on multiple other datasets and tasks, refer to the file [hiexpl_experiments.sh](hiexpl_experiments.sh).


## 2. For Feature Extraction Task
For performing the important phrase extraction for the feature extraction (NER) task,
follow the below steps,

### 2.1 Train the Feature Extraction (NER) Model (used in SOC)
To train the feature extraction (NER) model using SciBERT Entity Token Classifier model architecture, run the following command,

```
CUDA_VISIBLE_DEVICES=<gpu_num> python3 run_NER_classifier.py  --data_dir ../data/ner_dataset \
                                                              --output_dir ../models/repr_bert_ner_seq \
                                                              --model_name_or_path allenai/scibert_scivocab_uncased \
                                                              --tokenizer_name_or_path allenai/scibert_scivocab_uncased \
                                                              --num_epochs 5 \
                                                              --train_batch_size 32 \
                                                              --eval_batch_size 32
```

### 2.2 Train the Language Model (used for Context Sampling in SOC)
You can use the same language model trained for the reproducibility classification task.
As the language model is trained on the raw training domain corpus without labels, it is
independent of the target task.

### 2.3 Get Explanations using Sampling and Occlusion (SOC) Algorithm
To obtain the explanations using the Sampling and Occlusion (SOC) algorithm, use the 
below command,

```
export algo=soc
export exp_name=train_full
export model_path=repr_bert_ner_seq/model_best
export lm_path=repr_claims_lstm_lm_bert_tokenized

CUDA_VISIBLE_DEVICES=<gpu_num> python explain.py    --resume_snapshot models/${model_path} \
                                                    --method ${algo} \
                                                    --use_const_parse \
                                                    --lm_path models/${lm_path} \
                                                    --exp_name ${exp_name} \
                                                    --task repr_ner_seq \
                                                    --start 0 \
                                                    --stop -1 \
                                                    --target_entity_class ALL \
                                                    --explain_model bert \
                                                    --bert_model allenai/scibert_scivocab_uncased \
                                                    --use_bert_tokenizer \
                                                    --nb_range 10 \
                                                    --sample_n 20 \
                                                    --dataset train
```


**Note:** Run the above commands inside the [hiexpl_soc_ner/](hiexpl_soc_ner/) folder.

For a complete list of experiments on multiple other datasets and tasks, refer to the file [hiexpl_ner_experiments.sh](hiexpl_ner_experiments.sh).




