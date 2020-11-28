import pandas as pd
import json
from sklearn.metrics import mean_squared_error
import math

if __name__ == "__main__":
    inp1 = "../data_processed/ta2_reg_folds/fold_full/data.json"
    inp2 = "ta2_reg_checkpoint/outputs/TA2_M1_prediction_from_TA2REG.csv"

    df1 = pd.read_json(inp1)
    df2 = pd.read_csv(inp2)

    df1 = df1[["DOI_CR", "label", "Fold_Id"]]

    pid_pred_map = {}
    for idx, cur_row in df2.iterrows():
        pid = cur_row["paper_id"]
        pred = cur_row["pred_label"]
        pid_pred_map[pid] = pred

    df1["pred_label"] = df1.apply(lambda x: pid_pred_map[x["DOI_CR"]], axis=1)
    print(df1.columns)

    for cur_fold in range(1, 11):
        cur_gt = list(df1[df1["Fold_Id"] == cur_fold]["label"])
        cur_pred = list(df1[df1["Fold_Id"] == cur_fold]["pred_label"])
        rmse = math.sqrt(mean_squared_error(cur_gt, cur_pred))
        print("Fold id = ", cur_fold, " RMSE = ", rmse)
