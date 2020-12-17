# Train the BERT NER model
CUDA_VISIBLE_DEVICES=7 python3 run_NER_classifier.py
CUDA_VISIBLE_DEVICES=7 python3 run_NER_classifier.py

# SOC for NER SEQ
export algo=soc
export exp_name=repr_ner_seq_test_sample
export model_path=repr_bert_ner_seq/model_best
export lm_path=repr_claims_lstm_lm_bert_tokenized

# For entities SS, TN and SP respectively
CUDA_VISIBLE_DEVICES=4 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task repr_ner_seq --target_entity_class SS --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=4 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task repr_ner_seq --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=4 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task repr_ner_seq --target_entity_class SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test


# Train the BERT NER TC model
CUDA_VISIBLE_DEVICES=7 python3 run_NER_text_classifier.py

# SOC for NER TC
export algo=soc
export exp_name=repr_ner_tc_test_sample
export model_path=repr_bert_ner_tc/model_best
export lm_path=repr_claims_lstm_lm_bert_tokenized
CUDA_VISIBLE_DEVICES=7 python explain.py --resume_snapshot models/${model_path} --method ${algo} --lm_path models/${lm_path} --batch_size 1 --start 0 --stop 10 --exp_name ${exp_name} --task repr_ner_tc --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test






