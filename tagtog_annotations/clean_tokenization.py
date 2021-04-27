import os

def cleanData(input_path):
    result_str = ""
    with open(input_path, "r") as f:

        end_flag = True
        prev_split = ""
        for cur_line in f:
            cur_split = cur_line.split()

            # Empty line
            if len(cur_split) == 0:
                result_str += "\n"
                continue

            # Not properly tokenized token
            if len(cur_split[0]) >= 3 and cur_split[0].endswith("-") and cur_split[0][:-1]:
                end_flag = False
                prev_split = cur_split
                continue

            if end_flag:
                result_str += "\t".join(cur_split)
                result_str += "\n"
            else:
                new_split = [prev_split[0][:-1] + cur_split[0], prev_split[1]]
                result_str += "\t".join(new_split)
                result_str += "\n"
                end_flag = True
                prev_split = ""

    return result_str

if __name__ == "__main__":
    base_dir = "../data_processed/ner_dataset_v2"
    out_dir = "../data_processed/ner_dataset_v2_cleaned"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for type_path in ["test", "train"]:
        inp_path = os.path.join(base_dir, type_path + ".txt")
        out_path = os.path.join(out_dir, type_path + ".txt")

        cleaned_data = cleanData(inp_path)
        with open(out_path, "w") as f:
            f.write(cleaned_data)