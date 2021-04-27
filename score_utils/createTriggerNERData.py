import argparse
import json
import numpy as np
import os


def filterTriggers(triggers_list, topk=None, threshold=None):
    triggers_list_filtered = []
    cur_k = 0
    for cur_phrase in triggers_list:
        if cur_phrase["score"] >= threshold and cur_k < topk:
            triggers_list_filtered.append(cur_phrase)
            cur_k += 1

    return triggers_list_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_file", type=str, default="soc_outputs/repr_ner_seq_depparse_test_sample_TN.json",
                        help="Input file which contains the outputs from SOC algorithm")
    parser.add_argument("--out_dir", type=str, default="soc_outputs/repr_ner_seq_depparse_test_sample_TN",
                        help="Output directory to store the NER dataset in TriggerNER format")
    parser.add_argument("--soc_topk", type=int, default=1000,
                        help="Filter the trigger phrases to keep only the top-k triggers for each instance. Use very high top-k such as 1000 to avoid this filtering")
    parser.add_argument("--soc_threshold", type=float, default=-1000.0,
                        help="Filter the trigger phrases to keep only the triggers above a particular threshold of importance score for each instance. Use very low threshold such as -1000 to avoid this filtering")
    args = parser.parse_known_args()[0]

    with open(args.inp_file, "r") as f:
        soc_dict_list = json.load(f)

    result_str = ""
    for cur_dict in soc_dict_list:
        cur_sent = cur_dict["sentence"]

        # Use the extracted triggers only when the model prediction is same as ground truth prediction
        if cur_dict["predicted_label"] is not None:
            if type(cur_dict["predicted_label"]) is int:
                if cur_dict["label_idx"] == cur_dict["predicted_label"]:
                    triggers = filterTriggers(cur_dict["important_phrases"], args.soc_topk, args.soc_threshold)
                else:
                    triggers = []
                    print(cur_dict["label_idx"], cur_dict["predicted_label"])
            else:
                if cur_dict["label"] == cur_dict["predicted_label"]:
                    triggers = filterTriggers(cur_dict["important_phrases"], args.soc_topk, args.soc_threshold)
                else:
                    triggers = []
                    print(cur_dict["label"], cur_dict["predicted_label"])
        else:
            triggers = []

        cur_sent_seq = cur_sent.split()
        cur_lab_seq = np.array(["O"] * len(cur_sent_seq), dtype='object')

        # Mark the triggers
        for t_idx, cur_trigger in enumerate(triggers):
            cur_trig_span_start, cur_trig_span_end = cur_trigger["lm_span"][0], cur_trigger["lm_span"][1]
            cur_lab_seq[cur_trig_span_start: cur_trig_span_end + 1] = "T-" + str(t_idx)

        # Mark the entity span
        ent_label = cur_dict["label"]
        ent_span_start, ent_span_end = cur_dict["entity_span"][0], cur_dict["entity_span"][-1]
        cur_lab_seq[ent_span_start] = "B-" + ent_label
        cur_lab_seq[ent_span_start + 1: ent_span_end + 1] = "I-" + ent_label

        # Write to the output string
        for cur_token, cur_label in zip(cur_sent_seq, cur_lab_seq):
            result_str += cur_token + "\t" + cur_label + "\n"
        result_str += "\n"

    # Write to file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_file = os.path.join(args.out_dir, "trigger_100.txt")
    with open(out_file, "w") as f:
        f.write(result_str)
