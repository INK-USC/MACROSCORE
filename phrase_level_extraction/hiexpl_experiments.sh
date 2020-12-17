############ For the Repr dataset: (Updated code) ############


# 1. Classifier training: (done)
mkdir models
export model_path=repr_bert
CUDA_VISIBLE_DEVICES=5 python -m bert.run_classifier \
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

# 4. Visualize: (done)
python visualize.py --file outputs/repr/soc_bert_results/socrepr_demo.pkl --model repr_model --method bert



############ For the Repr Claims dataset: (Updated code) ############


# 1. Classifier training: (done)
mkdir models
export model_path=repr_claims_bert
CUDA_VISIBLE_DEVICES=1 python -m bert.run_classifier \
  --task_name repr \
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

# 2. LM training: (done)
export lm_path=repr_claims_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=2 python -m lm.lm_train --task repr --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer --batch_size 32 --epochs 40

# 3. SOC algorithm: (done)
export algo=soc
export exp_name=repr_claims_train
export model_path=repr_claims_bert
export lm_path=repr_claims_lstm_lm_bert_tokenized
mkdir repr
mkdir repr/results
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 703 --exp_name ${exp_name} --task repr --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset train

# 4. Post-hoc analysis:



############ For the Repr Segments dataset (sentence wise splitting): (Updated code) ############


# 1. Classifier training: (done)
mkdir models
export model_path=repr_segments_bert
CUDA_VISIBLE_DEVICES=4 python -m bert.run_classifier \
  --task_name repr \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/repr_segments \
  --bert_model allenai/scibert_scivocab_uncased \
  --max_seq_length 500 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --output_dir models/${model_path}

# 2. LM training: (done)
export lm_path=repr_segments_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=5 python -m lm.lm_train --task repr --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer --batch_size 32 --epochs 40

# 3. SOC algorithm: (ongoing)
export algo=soc
export exp_name=repr_segments_train_temp2
export model_path=repr_segments_bert
export lm_path=repr_segments_lstm_lm_bert_tokenized_old
CUDA_VISIBLE_DEVICES=4 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 5 --exp_name ${exp_name} --task repr --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset train

# 4. Post-hoc analysis:


############ For the Repr Sentences dataset (bilstm): (Updated code) ############


# 1. Classifier training: (done)
mkdir models
export model_path=repr_sentences_bert
CUDA_VISIBLE_DEVICES=1 python -m bert.run_classifier \
  --task_name repr \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data/repr_sentences \
  --bert_model allenai/scibert_scivocab_uncased \
  --max_seq_length 500 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --output_dir models/${model_path}

# 2. LM training: (done)
export lm_path=repr_sentences_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=2 python -m lm.lm_train --task repr --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer --batch_size 32 --epochs 40
CUDA_VISIBLE_DEVICES=2 python -m lm.lm_train --task repr --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer --batch_size 32 --epochs 40

# 3. SOC algorithm: (ongoing)
export algo=soc
export exp_name=repr_sentences_full_v1
export model_path=repr_sentences_bert
export lm_path=repr_sentences_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 400 --exp_name ${exp_name} --task repr --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset train


export algo=soc
export exp_name=repr_sentences_full_v2
export model_path=repr_sentences_bert
export lm_path=repr_sentences_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 400 --stop 900 --exp_name ${exp_name} --task repr --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset train

# 4. Post-hoc analysis:



############ For the Yelp dataset: (Old code) ############


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


############ For the SST-2 dataset: (Old code) ############


# 1. Classifier training:
mkdir models
export model_path=sst_bert
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

# 2. LM training:
export lm_path=yelp_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=0 python -m lm.lm_train --task yelp --save_path models/${lm_path} --lr 0.0002 --use_bert_tokenizer

# 3. SOC algorithm:
export algo=soc
export exp_name=yelp_demo
export model_path=yelp_bert
export lm_path=yelp_lstm_lm_bert_tokenized
mkdir yelp
mkdir yelp/results
CUDA_VISIBLE_DEVICES=0 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task yelp --explain_model bert --use_bert_tokenizer --nb_range 10 --sample_n 20

#
export exp_name=yelp_agg_demo
CUDA_VISIBLE_DEVICES=0 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task yelp --explain_model bert --use_bert_tokenizer --nb_range 10 --sample_n 20 --agg

# 4. Visualize:
python visualize.py --file outputs/yelp/soc_bert_results/socyelp_demo.pkl --model yelp_model --method bert











