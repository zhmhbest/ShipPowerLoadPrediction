import torch
from torch.nn.modules.module import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import RNN
from torch.nn.modules.activation import ReLU, Sigmoid


class BPModule(Module):
    def __init__(self, input_size: int = 9, output_size: int = 1):
        super().__init__()
        self.linear1 = Linear(input_size, 16)
        self.bp1 = Linear(16, 64)
        self.bp2 = Linear(64, 128)
        self.bp3 = Linear(128, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, output_size)
        # self.sigmoid = Sigmoid()
        # self.linear4 = Linear(output_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.bp1(x)
        x = self.bp2(x)
        x = self.bp3(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = self.sigmoid(x)
        # x = self.linear4(x)
        return x


if __name__ == '__main__':
    x_demo = torch.randn(32, 12, 9)
    y_demo = torch.randn(32, 12, 1)
    model = BPModule(input_size=9, output_size=1)
    with torch.no_grad():
        y = model(x_demo)
        print(y.shape)
