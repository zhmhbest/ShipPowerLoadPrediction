from tqdm import tqdm
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from CustomDataset import PowerLoadDataset, DataLoader
#
from ContrastLSTMModule import LSTMModule
from ContrastRNNModule import RNNModule
from ContrastBPModule import BPModule
#
from CustomDevice import DEVICE
from CustomEvaluate import rmse_nd, mape_nd
from CustomModelName import get_model_name


# LSTMModule RNNModule BPModule
MODEL_NAME = "BPModule"


if __name__ == '__main__':
    dataset = PowerLoadDataset(time_step=32, data_type="test")
    data_loader = DataLoader(dataset, batch_size=256)

    if "LSTMModule" == MODEL_NAME:
        model = LSTMModule(input_size=9, output_size=1).to(DEVICE)
    elif "RNNModule" == MODEL_NAME:
        model = RNNModule(input_size=9, output_size=1).to(DEVICE)
    else:
        model = BPModule(input_size=9, output_size=1).to(DEVICE)

    # 加载训练历史
    checkpoint = torch.load(get_model_name(MODEL_NAME, 300))
    model.load_state_dict(checkpoint['model'])

    y_pred_buff = []
    y_true_buff = []
    with torch.no_grad():
        for i, (x_batch, y_batch) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            # GPU
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            #
            y_predict: torch.Tensor = model(x_batch)
            y_pred_buff.append(y_predict.detach().cpu().numpy())
            y_true_buff.append(y_batch.detach().cpu().numpy())

    y_pred = np.vstack(y_pred_buff).reshape((-1, 1))
    y_true = np.vstack(y_true_buff).reshape((-1, 1))
    #
    rmse_val = rmse_nd(y_pred, y_true)
    mape_val = mape_nd(y_pred, y_true)
    print("RMSE", rmse_val)
    print("MAPE", mape_val)
    # 还原数据
    y_pred = dataset.inverse_output(y_pred)
    y_true = dataset.inverse_output(y_true)
    # 绘图
    matplotlib.use('TkAgg')
    plt.plot(y_pred[200:500], label="Prediction")
    plt.plot(y_true[200:500], label="Truth")
    plt.title(MODEL_NAME)
    plt.legend()
    plt.show()
