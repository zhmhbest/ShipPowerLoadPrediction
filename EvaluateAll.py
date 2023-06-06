import os
# from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib
import matplotlib.pyplot as plt
#
from sklearn.svm import SVR
from ContrastLSTMModule import LSTMModule
from ContrastRNNModule import RNNModule
from ContrastBPModule import BPModule
#
from CustomDataset import PowerLoadDataset, DataLoader
from CustomEvaluate import rmse_nd, mape_nd
from CustomModelName import get_model_name
from CustomDevice import DEVICE
#
matplotlib.use('TkAgg')


if __name__ == '__main__':
    # 加载数据集
    dataset = PowerLoadDataset(time_step=32, data_type="test")
    data_loader = DataLoader(dataset, batch_size=int(500 / 32))

    # 加载所有模型
    with open(get_model_name("SVM", 0), "rb") as fd:
        model_svm: SVR = pickle.load(fd)
    model_lstm = LSTMModule(input_size=9, output_size=1).to(DEVICE)
    model_lstm.load_state_dict(torch.load(get_model_name("LSTMModule", 300))['model'])
    #
    model_rnn = RNNModule(input_size=9, output_size=1).to(DEVICE)
    model_rnn.load_state_dict(torch.load(get_model_name("RNNModule", 300))['model'])
    #
    model_bp = BPModule(input_size=9, output_size=1).to(DEVICE)
    model_bp.load_state_dict(torch.load(get_model_name("BPModule", 300))['model'])

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader, 0):
            # nd
            x_nd: np.ndarray = x_batch.numpy()
            y_nd: np.ndarray = y_batch.numpy()
            # GPU
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            #
            y_true: np.ndarray = y_nd.reshape(-1)
            y_lstm: np.ndarray = model_lstm(x_batch).detach().cpu().numpy().reshape(-1)
            y_rnn: np.ndarray = model_rnn(x_batch).detach().cpu().numpy().reshape(-1)
            y_bp: np.ndarray = model_bp(x_batch).detach().cpu().numpy().reshape(-1)
            y_svm: np.ndarray = model_svm.predict(x_nd.reshape((-1, 9)))
            # 还原数据
            y_true = dataset.inverse_output(y_true)
            y_lstm = dataset.inverse_output(y_lstm)
            y_rnn = dataset.inverse_output(y_rnn)
            y_bp = dataset.inverse_output(y_bp)
            y_svm = dataset.inverse_output(y_svm)
            # 存储数据
            pd.DataFrame({
                "TRUE": y_true,
                "LSTM": y_lstm,
                "RNN": y_rnn,
                "BP": y_bp,
                "SVM": y_svm,
            }).to_csv("./checkpoint/predict.csv", index=False)
            #
            print(pd.DataFrame([
                [rmse_nd(y_lstm, y_true), mape_nd(y_lstm, y_true)],
                [rmse_nd(y_rnn, y_true), mape_nd(y_rnn, y_true)],
                [rmse_nd(y_bp, y_true), mape_nd(y_bp, y_true)],
                [rmse_nd(y_svm, y_true), mape_nd(y_svm, y_true)],
            ],
                columns=["RMSE", "MAPE"],
                index=["LSTM", "RNN", "BP", "SVM"]
            ))
            break
