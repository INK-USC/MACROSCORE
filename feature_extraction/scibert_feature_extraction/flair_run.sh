# 7: only word-embeddings, 8: use char-embeddings, 9: use flair-embeddings
# Current Best: 7
CUDA_VISIBLE_DEVICES=2 python flair_train.py --folder data_processed --idx 9

CUDA_VISIBLE_DEVICES=1 python flair_eval.py --folder score_7 --method chunk \
| tee logs/flair_eval_7_chunk.log

