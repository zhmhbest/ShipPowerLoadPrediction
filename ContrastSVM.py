import os
import numpy as np
from sklearn.svm import SVR  # 回归
import pickle
import matplotlib
import matplotlib.pyplot as plt
from CustomDataset import PowerLoadDataset
from CustomEvaluate import rmse_nd, mape_nd
from CustomModelName import get_model_name

if __name__ == '__main__':
    model_name = get_model_name("SVM", 0)
    dataset = PowerLoadDataset(time_step=0, data_type="")

    x_train = dataset.x_train
    y_train = dataset.y_train.reshape(-1)
    x_test = dataset.x_test
    y_test = dataset.y_test.reshape(-1)

    if os.path.exists(model_name):
        print("Loading...")
        with open(model_name, "rb") as fd:
            model: SVR = pickle.load(fd)
    else:
        print("Training...")
        model = SVR()
        model.fit(x_train, y_train)
        with open(model_name, "wb") as fd:
            pickle.dump(model, fd)

    y_pred = model.predict(x_test)
    rmse_val = rmse_nd(y_pred, y_test)
    mape_val = mape_nd(y_pred, y_test)
    print("RMSE", rmse_val)
    print("MAPE", mape_val)
    # 还原数据
    y_pred = dataset.inverse_output(y_pred)
    y_true = dataset.inverse_output(y_test)
    # 绘图
    matplotlib.use('TkAgg')
    plt.plot(y_pred[200:500], label="Prediction")
    plt.plot(y_true[200:500], label="Truth")
    plt.title("SVM")
    plt.legend()
    plt.show()
