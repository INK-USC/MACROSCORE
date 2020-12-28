import pandas as pd
import os
from sklearn.model_selection import KFold

if __name__ == "__main__":
    inp_path = "notebooks/data/TA2_classify_data_final_with_imp_sentences_bilstm_only.json"
    df = pd.read_json(inp_path)
    df = df[["paper_id", "important_segment", "label", "Fold_Id"]]

    print(df.columns)

    base_out_dir = "repr_sentences"
    base_fold_dir = "repr_sentences_5fold"

    # Write the full data:
    if not os.path.exists(base_out_dir):
        os.makedirs(base_out_dir)

    out1 = os.path.join(base_out_dir, "full_data.csv")
    df.to_csv(out1, index=False)

    # Write the folds
    if not os.path.exists(base_fold_dir):
        os.makedirs(base_fold_dir)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    idx = 1
    for train_idx, test_idx in kf.split(df):
        df_train = df.iloc[list(train_idx)]
        df_test = df.iloc[list(test_idx)]

        folder_path = os.path.join(base_fold_dir, "fold_" + str(idx))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_train.to_csv(os.path.join(folder_path, "train.csv"), index=False)
        df_test.to_csv(os.path.join(folder_path, "test.csv"), index=False)
        df_test.to_csv(os.path.join(folder_path, "dev.csv"), index=False)
        idx += 1





