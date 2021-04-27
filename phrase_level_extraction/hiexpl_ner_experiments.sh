# Train the BERT NER model
CUDA_VISIBLE_DEVICES=7 python3 run_NER_classifier.py
CUDA_VISIBLE_DEVICES=1 python3 run_NER_classifier.py

# SOC for NER SEQ
export algo=soc
export exp_name=test_sample
export model_path=repr_bert_ner_seq/model_best
export lm_path=repr_claims_lstm_lm_bert_tokenized

# For entities SS, TN and SP respectively (NER SEQ, Const parse)
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 10 --target_entity_class SS --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 10 --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 10 --target_entity_class SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop -1 --target_entity_class SS,TN,SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop -1 --target_entity_class SS,TN,SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset train
# --start 0 --stop 10

# For entities SS, TN and SP respectively (NER SEQ, Dep parse)
CUDA_VISIBLE_DEVICES=5 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 2 --target_entity_class SS --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=5 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 2 --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 2 --target_entity_class SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop -1 --target_entity_class SS,TN,SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=1 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop -1 --target_entity_class SS,TN,SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset train
# --start 0 --stop 10

CUDA_VISIBLE_DEVICES=5 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_seq --start 0 --stop 10 --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test


################################################################

# Train the BERT NER TC model
CUDA_VISIBLE_DEVICES=7 python3 run_NER_text_classifier.py

# SOC for NER TC
export algo=soc
export exp_name=test_sample
export model_path=repr_bert_ner_tc/model_best
export lm_path=repr_claims_lstm_lm_bert_tokenized

# For entities SS, TN and SP respectively (NER TC, Const parse)
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class SS --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_const_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class SS,TN,SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
# --start 0 --stop 10

# For entities SS, TN and SP respectively (NER TC, Dep parse)
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class SS --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
CUDA_VISIBLE_DEVICES=2 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 10 --target_entity_class SS,TN,SP --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test
# --start 0 --stop 10




CUDA_VISIBLE_DEVICES=5 python explain.py --resume_snapshot models/${model_path} --method ${algo} --use_dep_parse --lm_path models/${lm_path} --exp_name ${exp_name} --task repr_ner_tc --start 0 --stop 2 --target_entity_class TN --explain_model bert  --bert_model allenai/scibert_scivocab_uncased --use_bert_tokenizer --nb_range 10 --sample_n 20 --dataset test

