
# EXP 1
# 1: lower case, 2: use loss, 3: use score, 4: use char-embeddings, 5: use flair-embeddings
# 6: same as 3, train on all
# Current Best: 3
# CUDA_VISIBLE_DEVICES=0 python flair_train.py --folder data_processed --idx 6

# EXP 2
# 7: only word-embeddings, 8: use char-embeddings, 9: use flair-embeddings
# Current Best: 7
CUDA_VISIBLE_DEVICES=2 python flair_train.py --folder data_processed --idx 9

CUDA_VISIBLE_DEVICES=1 python flair_eval.py --folder score_7 --method chunk \
| tee logs/flair_eval_7_chunk.log

