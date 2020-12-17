import os
import json, jsonlines

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

    return instance_list

def createDataset(inp_file):
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
                cur_label = cur_line.split()[-1]
                cur_sentence_seq.append(cur_input)
                cur_label_seq.append(cur_label)
            except Exception as e:
                pass

    # Encode the labels
    label_list = ['SP', 'TE', 'SD', 'ES', 'PR', 'PV', 'SS', 'TN']
    label2idx = {}
    for i, label in enumerate(label_list):
        label2idx[label] = i

    for idx in range(len(result)):
        result[idx]["label_idx"] = label2idx[result[idx]["label"]]

    return result

if __name__ == "__main__":

    base_dir = "../data/ner_dataset/"
    train_file = os.path.join(base_dir, "train.txt")
    test_file = os.path.join(base_dir, "test.txt")

    train_dataset = createDataset(train_file)
    test_dataset = createDataset(test_file)

    out_dir = "../data/ner_dataset_tc/"
    train_out = os.path.join(out_dir, "train.txt")
    test_out = os.path.join(out_dir, "test.txt")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with jsonlines.open(train_out, "w") as f:
        f.write_all(train_dataset)

    with jsonlines.open(test_out, "w") as f:
        f.write_all(test_dataset)





