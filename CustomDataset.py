import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class PowerLoadDataset(Dataset):
    def __init__(self, time_step: int, data_type: str) -> None:
        super(PowerLoadDataset, self).__init__()
        data_loc = "./dataset/data.csv"
        df: pd.DataFrame = pd.read_csv(data_loc)

        # 剪裁余数（构建time_step的准备工作）
        if time_step > 0:
            raw_sum_size = df.shape[0]
            need_del_size = raw_sum_size % time_step
            tail_pos = raw_sum_size - 1 - need_del_size
            df = df.loc[0:tail_pos]
            sum_size = df.shape[0]

        # 切割 X Y
        x_df = df[[
            "year", "month", "day", "weekday", "hour",
            "temperature", "is_weekday", "season", "festival"]]
        y_df = df[["demand"]]
        #
        x_raw_nd: np.ndarray = x_df.to_numpy()
        y_raw_nd: np.ndarray = y_df.to_numpy()

        # 数据预处理 StandardScaler MinMaxScaler
        x_scaler = StandardScaler()
        x_nd = x_scaler.fit_transform(x_raw_nd)
        self.x_scaler = x_scaler
        #
        y_scaler0 = StandardScaler()
        y_scaler1 = MinMaxScaler()
        y_nd = y_scaler0.fit_transform(y_raw_nd)
        y_nd = y_scaler1.fit_transform(y_nd)
        self.y_scaler0 = y_scaler0
        self.y_scaler1 = y_scaler1

        # 构建time_step
        if time_step > 0:
            x_dim = x_nd.shape[-1]
            y_dim = y_nd.shape[-1]
            x_nd = x_nd.reshape((-1, time_step, x_dim))
            y_nd = y_nd.reshape((-1, time_step, y_dim))

        # 切割数据集
        x_train, x_test, y_train, y_test = train_test_split(
            x_nd, y_nd, test_size=0.30, random_state=None, shuffle=False, stratify=None)
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.y_test = y_test.astype(np.float32)
        if "train" == data_type:
            self.x = torch.from_numpy(self.x_train)
            self.y = torch.from_numpy(self.y_train)
        else:
            self.x = torch.from_numpy(self.x_test)
            self.y = torch.from_numpy(self.y_test)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, _i) -> any:
        return self.x[_i], self.y[_i]

    def inverse_output(self, y: np.ndarray):
        return self.y_scaler0.inverse_transform(
            self.y_scaler1.inverse_transform(y.reshape((-1, 1)))
        ).reshape(-1)


if __name__ == '__main__':
    data_loader = DataLoader(PowerLoadDataset(time_step=12, data_type="test"), batch_size=32)

    for i, (x_batch, y_batch) in enumerate(data_loader, 0):
        print(i, x_batch.shape, y_batch.shape)
        break
