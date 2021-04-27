import json
import argparse
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", type=str, default="soc_outputs/repr_ner_seq_constparse/",
                        help="Input directory which contains the outputs from SOC algorithm")
    parser.add_argument("--out_file", type=str, default="soc_outputs/repr_ner_seq_constparse_trees_train_full_ALL.json",
                        help="Output file to store the combined SOC output")

    args = parser.parse_known_args()[0]
    print(args)

    files_list = os.listdir(args.inp_dir)
    total_data = []
    for cur_file in files_list:
        if cur_file.endswith(".json"):
            cur_file_path = os.path.join(args.inp_dir, cur_file)

            with open(cur_file_path, "r") as f:
                cur_data = json.load(f)
                total_data += cur_data

    random.shuffle(total_data)

    out_file_path = os.path.join(args.inp_dir, args.out_file)
    with open(out_file_path, "w") as f:
        json.dump(total_data, f)










