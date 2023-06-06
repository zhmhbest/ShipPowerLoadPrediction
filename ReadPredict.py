import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from CustomEvaluate import rmse_nd, mape_nd
matplotlib.use('TkAgg')


if __name__ == '__main__':
    df = pd.read_csv("./checkpoint/predict.csv")
    y_true: np.ndarray = df["TRUE"]
    y_lstm: np.ndarray = df["LSTM"]
    y_rnn: np.ndarray = df["RNN"]
    y_bp: np.ndarray = df["BP"]
    y_svm: np.ndarray = df["SVM"]
    #
    evaluate = pd.DataFrame([
        [rmse_nd(y_lstm, y_true), mape_nd(y_lstm, y_true)],
        [rmse_nd(y_rnn, y_true), mape_nd(y_rnn, y_true)],
        [rmse_nd(y_bp, y_true), mape_nd(y_bp, y_true)],
        [rmse_nd(y_svm, y_true), mape_nd(y_svm, y_true)],
    ],
        columns=["RMSE", "MAPE"],
        index=["LSTM", "RNN", "BP", "SVM"]
    )
    evaluate.to_csv("./checkpoint/evaluate.csv")
    print(evaluate)
    # 绘图
    plt.plot(y_true, label="Truth", linestyle="-")
    plt.plot(y_lstm, label="LSTM", linestyle=":", marker=".")
    plt.plot(y_rnn, label="RNN", linestyle=":", marker="+")
    plt.plot(y_bp, label="BP", linestyle=":", marker="1")
    plt.plot(y_svm, label="SVM", linestyle=":", marker="*")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.grid()
    plt.show()
