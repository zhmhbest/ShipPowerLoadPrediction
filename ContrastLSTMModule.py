import torch
from torch.nn.modules.module import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.activation import ReLU, Sigmoid


class LSTMModule(Module):
    def __init__(self, input_size: int = 9, output_size: int = 1):
        super().__init__()
        self.linear1 = Linear(input_size, 16)
        self.lstm1 = LSTM(input_size=16, hidden_size=64, num_layers=3, batch_first=True, dropout=0.2)
        self.lstm2 = LSTM(input_size=64, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2)
        self.lstm3 = LSTM(input_size=128, hidden_size=64, num_layers=3, batch_first=True, dropout=0.2)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, output_size)
        # self.sigmoid = Sigmoid()
        # self.linear4 = Linear(output_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        (x, _) = self.lstm1(x)
        (x, _) = self.lstm2(x)
        (x, _) = self.lstm3(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = self.sigmoid(x)
        # x = self.linear4(x)
        return x


if __name__ == '__main__':
    x_demo = torch.randn(32, 12, 9)
    y_demo = torch.randn(32, 12, 1)
    model = LSTMModule(input_size=9, output_size=1)
    with torch.no_grad():
        y = model(x_demo)
        print(y.shape)
