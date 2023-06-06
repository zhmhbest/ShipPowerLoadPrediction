import torch
from torch.nn.modules.rnn import LSTM


if __name__ == '__main__':
    x_demo = torch.randn(32, 12, 9)
    y_demo = torch.randn(32, 12, 1)
    lstm = LSTM(input_size=9, hidden_size=25, num_layers=6)
    with torch.no_grad():
        (y, (h, c)) = lstm(x_demo)
        print(y.shape)
        print(h.shape)
        print(c.shape)
