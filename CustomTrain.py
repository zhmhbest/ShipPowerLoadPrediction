import os
from tqdm import tqdm
import torch
from torch.optim.adam import Adam
from torch.nn.modules.loss import MSELoss
#
from CustomDataset import PowerLoadDataset, DataLoader
#
from ContrastLSTMModule import LSTMModule
from ContrastRNNModule import RNNModule
from ContrastBPModule import BPModule
#
from CustomDevice import DEVICE
from CustomModelName import get_model_name

# LSTMModule RNNModule BPModule
MODEL_NAME = "LSTMModule"
LAST_EPOCH_END = 100
CUR_EPOCH_NUM = 350


if __name__ == '__main__':
    data_loader = DataLoader(PowerLoadDataset(time_step=32, data_type="train"), batch_size=256)
    if "LSTMModule" == MODEL_NAME:
        model = LSTMModule(input_size=9, output_size=1).to(DEVICE)
    elif "RNNModule" == MODEL_NAME:
        model = RNNModule(input_size=9, output_size=1).to(DEVICE)
    else:
        model = BPModule(input_size=9, output_size=1).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = MSELoss()

    # 加载训练历史
    if 0 != LAST_EPOCH_END:
        print("Load last model")
        checkpoint = torch.load(get_model_name(MODEL_NAME, LAST_EPOCH_END))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 本轮训练
    for epoch in range(1 + LAST_EPOCH_END, 1 + LAST_EPOCH_END + CUR_EPOCH_NUM):
        loss_item_sum: int = 0
        loss_running: float = 0.0
        for i, (x_batch, y_batch) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            # GPU
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # 前向传播
            optimizer.zero_grad()
            y_predict = model(x_batch)
            # print(y_predict.shape)

            # 计算损失
            loss_val = criterion(y_predict, y_batch)

            # 反向传播及优化
            loss_val.backward()
            optimizer.step()

            loss_running += loss_val.item() * x_batch.shape[0]
            loss_item_sum += x_batch.shape[0]
        loss_running /= loss_item_sum
        print(f"epoch:{epoch}, loss:{loss_running}")

        # 每10次存一次模型
        if 0 == (epoch % 25):
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, get_model_name(MODEL_NAME, epoch, f"loss={loss_running}"))
