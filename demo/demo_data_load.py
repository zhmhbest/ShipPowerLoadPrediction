import pandas as pd
import numpy as np
import torch

if __name__ == '__main__':
    # data_loc = "./dataset/PDB_Load_History.csv"
    data_loc = "./dataset/data.csv"
    df: pd.DataFrame = pd.read_csv(data_loc)
    print(df.dtypes)
    # num              int64
    # year             int64
    # month            int64
    # day              int64
    # weekday          int64
    # hour             int64
    # temperature      int64
    # is_weekday       int64
    # season           int64
    # festival         int64
    # demand         float64
    df_x = df[[
        "year", "month", "day", "weekday", "hour",
        "temperature", "is_weekday", "season", "festival"]]
    df_y = df[["demand"]]

    nd_x: np.ndarray = df_x.to_numpy()
    nd_y: np.ndarray = df_y.to_numpy()

    tr_x = torch.from_numpy(nd_x.astype(np.float32))
    tr_y = torch.from_numpy(nd_y.astype(np.float32))

    print()
