import json

if __name__ == "__main__":
    inp_path = "repr_claims_results/socrepr_claims_train.json"
    out_path = "repr_claims_results/socrepr_claims_train_filtered.json"

    with open(inp_path, "r") as f:
        data = json.load(f)

    data_filtered = []
    for cur_row in data:
        if cur_row["label"] == cur_row["predicted_label"] and cur_row["label"] == 1:
            data_filtered.append(cur_row)

    print(len(data_filtered))
    with open(out_path, "w") as f:
        json.dump(data_filtered, f, indent=2)

