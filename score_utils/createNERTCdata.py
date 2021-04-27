import os
import json, jsonlines
import argparse

def getInstances(cur_sentence_seq, cur_label_seq):
    instance_list = []
    sentence = " ".join(cur_sentence_seq)

    start_idx = None
    end_idx = None
    tag = None
    for idx, cur_label in enumerate(cur_label_seq):
        if cur_label.startswith("B-"):
            if start_idx is not None and end_idx is not None and tag is not None:
                cur_dict = {
                    "sentence": sentence,
                    "entity": " ".join(cur_sentence_seq[start_idx: end_idx + 1]),
                    "entity_span": list(range(start_idx, end_idx + 1, 1)),
                    "label": tag,
                }
                instance_list.append(cur_dict)

            start_idx = idx
            end_idx = idx
            tag = cur_label.split("-")[-1]
        elif tag is not None and cur_label.startswith("I-") and cur_label.split("-")[-1] == tag:
            end_idx = idx
        elif start_idx is not None and end_idx is not None and tag is not None:
            cur_dict = {
                "sentence": sentence,
                "entity": " ".join(cur_sentence_seq[start_idx: end_idx + 1]),
                "entity_span": list(range(start_idx, end_idx + 1, 1)),
                "label": tag,
            }
            start_idx = None
            end_idx = None
            tag = None
            instance_list.append(cur_dict)

    # Accounting when the entity is present at the end of the sentence
    if start_idx is not None and end_idx is not None and tag is not None:
        cur_dict = {
            "sentence": sentence,
            "entity": " ".join(cur_sentence_seq[start_idx: end_idx + 1]),
            "entity_span": list(range(start_idx, end_idx + 1, 1)),
            "label": tag,
        }
        start_idx = None
        end_idx = None
        tag = None
        instance_list.append(cur_dict)

    return instance_list

def createDataset(inp_file, label_col_index, label2idx):
    result = []

    with open(inp_file, "r") as f:
        data_lines = f.readlines()

    # Used for debugging (Set -1 to read the entire dataset)
    num_sentences_to_read = -1

    cur_sentence_seq = []
    cur_label_seq = []
    cur_sent_num = 0
    for cur_line in data_lines:
        cur_line = cur_line.strip()
        if len(cur_line) == 0 or cur_line.startswith('-DOCSTART'):
            # End of a sentence
            if len(cur_sentence_seq) != 0:
                result += getInstances(cur_sentence_seq, cur_label_seq)

                cur_sent_num += 1
                if num_sentences_to_read != -1 and cur_sent_num >= num_sentences_to_read:
                    break

            # Reset the sequences
            cur_sentence_seq = []
            cur_label_seq = []
        else:

            try:
                cur_input = cur_line.split()[0]
                cur_label = cur_line.split()[label_col_index]
                cur_sentence_seq.append(cur_input)
                cur_label_seq.append(cur_label)
            except Exception as e:
                pass

    for idx in range(len(result)):
        result[idx]["label_idx"] = label2idx[result[idx]["label"]]

    return result

def getLabels(inp_file, label_col_index):
    with open(inp_file, "r") as f:
        data_lines = f.readlines()

    label_set = set()
    for cur_line in data_lines:
        cur_line = cur_line.strip()
        if len(cur_line) == 0 or cur_line.startswith('-DOCSTART'):
            # End of a sentence
            continue
        else:
            try:
                cur_label_split = cur_line.split()[label_col_index].split("-")
                if len(cur_label_split) == 2:
                    label_set.add(cur_label_split[-1])
            except Exception as e:
                pass

    label_set = list(sorted(list(label_set)))

    # Encode the labels
    label2idx = {}
    for i, label in enumerate(label_set):
        label2idx[label] = i

    return label2idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../phrase_level_extraction/hiexpl_soc_ner/data/ner_dataset/",
                        help="NER Data directory (with train.txt and test.txt files)")
    parser.add_argument("--out_dir", type=str, default="../phrase_level_extraction/hiexpl_soc_ner/data/ner_dataset_tc/",
                        help="NER-TC Output directory (with train.jsonl and test.jsonl files)")
    parser.add_argument("--label_col_index", type=int, default=-1, help="Column index of the NER label in the dataset")
    args = parser.parse_known_args()[0]

    base_dir = args.data_dir
    train_file = os.path.join(base_dir, "train.txt")
    test_file = os.path.join(base_dir, "test.txt")

    label2idx = getLabels(train_file, label_col_index=args.label_col_index)
    train_dataset = createDataset(train_file, label_col_index=args.label_col_index, label2idx=label2idx)
    test_dataset = createDataset(test_file, label_col_index=args.label_col_index, label2idx=label2idx)

    out_dir = args.out_dir
    train_out = os.path.join(out_dir, "train.jsonl")
    test_out = os.path.join(out_dir, "test.jsonl")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with jsonlines.open(train_out, "w") as f:
        f.write_all(train_dataset)

    with jsonlines.open(test_out, "w") as f:
        f.write_all(test_dataset)





