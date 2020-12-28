import pandas as pd
from sklearn.model_selection import KFold
import os

if __name__ == "__main__":
    inp_data_path = "../hiexpl_soc_only/data/repr/repr_data.csv"
    base_out_dir = "../hiexpl_soc_only/data/repr_5fold/"
    df = pd.read_csv(inp_data_path)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    idx = 1
    for train_idx, test_idx in kf.split(df):
        df_train = df.iloc[list(train_idx)]
        df_test = df.iloc[list(test_idx)]

        folder_path = os.path.join(base_out_dir, "fold_" + str(idx))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_train.to_csv(os.path.join(folder_path, "train.csv"), index=False)
        df_test.to_csv(os.path.join(folder_path, "test.csv"), index=False)
        df_test.to_csv(os.path.join(folder_path, "dev.csv"), index=False)
        idx += 1


