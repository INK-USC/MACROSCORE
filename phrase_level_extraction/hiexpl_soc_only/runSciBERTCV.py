import os

if __name__ == "__main__":

    num_splits = 5
    gpu_device = 4
    base_command = """CUDA_VISIBLE_DEVICES={} python -m bert.run_classifier \
          --task_name repr \
          --do_train \
          --do_eval \
          --do_lower_case \
          --data_dir data/repr_segments_5fold/fold_{} \
          --bert_model allenai/scibert_scivocab_uncased \
          --max_seq_length 500 \
          --train_batch_size 8 \
          --learning_rate 2e-5 \
          --num_train_epochs 20 \
          --output_dir models/repr_segments_bert_CV_{}"""

    for i in range(1, num_splits+1):
        cur_command = base_command.format(gpu_device, i, i)
        print(cur_command)
        os.system(cur_command)