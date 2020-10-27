
# Run LM finetuning:
CUDA_VISIBLE_DEVICES=3 python3 -m lm.lm_train --task repr --save_path models/lm_ckpt --lr 0.0002 --use_bert_tokenizer --batch_size 1 --epochs 10

# Train the classifier:
python -m bert.run_classifier \
  --task_name repr \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir bert/data/repr \
  --bert_model allenai/scibert_scivocab_uncased \
  --max_seq_length 512 \
  --train_batch_size 1 \
  --learning_rate 1e-3 \
  --num_train_epochs 10 \
  --output_dir bert/models_repr


# For the Yelp dataset:

# 1. Classifier training: (done)
mkdir models
export model_path=yelp_bert
CUDA_VISIBLE_DEVICES=0 python -m bert.run_classifier \
  --task_name yelp \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/yelp_review_polarity_csv \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir models/${model_path}

# 2. LM training: (done)
export lm_path=yelp_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=0 python -m lm.lm_train --task yelp --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer

# 3. SOC algorithm: (done)
export algo=soc
export exp_name=yelp_demo
export model_path=yelp_bert
export lm_path=yelp_lstm_lm_bert_tokenized
mkdir yelp
mkdir yelp/results
CUDA_VISIBLE_DEVICES=0 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task yelp --explain_model bert --use_bert_tokenizer --nb_range 10 --sample_n 20

# (not working)
export exp_name=yelp_agg_demo
CUDA_VISIBLE_DEVICES=0 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task yelp --explain_model bert --use_bert_tokenizer --nb_range 10 --sample_n 20 --agg

# 4. Visualize: (not working)
python visualize.py --file outputs/yelp/soc_bert_results/socyelp_demo.pkl --model yelp_model --method bert


# For the Repr dataset: (Updated code)

# 1. Classifier training: (done)
mkdir models
export model_path=repr_bert
CUDA_VISIBLE_DEVICES=0 python -m bert.run_classifier \
  --task_name repr \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/repr \
  --bert_model allenai/scibert_scivocab_uncased \
  --max_seq_length 500 \
  --train_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir models/${model_path}

# 2. LM training: (done)
export lm_path=repr_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=0 python -m lm.lm_train --task repr --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer --batch_size 1 --epochs 10

# 3. SOC algorithm: (done)
export algo=soc
export exp_name=repr_demo
export model_path=repr_bert
export lm_path=repr_lstm_lm_bert_tokenized
mkdir repr
mkdir repr/results
CUDA_VISIBLE_DEVICES=0 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 5 --exp_name ${exp_name} --task repr --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20

# (done)
export algo=soc
export exp_name=repr_agg_demo
export model_path=repr_bert
export lm_path=repr_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=0 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 5 --exp_name ${exp_name} --task repr --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --agg

# 4. Visualize: (not working)
python visualize.py --file outputs/repr/soc_bert_results/socrepr_demo.pkl --model repr_model --method bert












