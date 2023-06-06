from abc import abstractmethod
from typing import List, Iterator, Tuple

import numpy as np
from numpy import ndarray


class Layer:
    def __init__(self):
        # 是否保留衍生变量（请在forward时判断）
        self.retain_derived = True
        # 衍生变量列表
        self.derivations = {'X': []}

    def define_derivation(self, key):
        """定义衍生变量"""
        self.derivations[key] = []

    def clear_derivations(self):
        """清空衍生变量记录"""
        for key in self.derivations:
            self.derivations[key].clear()

    def iterable_pl_pz_x(self, pl_pz_list: List[ndarray]) -> Iterator[Tuple[ndarray, ndarray]]:
        """
        在定义了衍生变量X的情况下，返回可遍历的列表(pl_pz, x)
        """
        if 'X' not in self.derivations.keys():
            raise ValueError("Undefined 'X' in derivations.")
        x_list = self.derivations['X']
        if len(x_list) != len(pl_pz_list):
            raise BufferError("The derived quantity does not match the partial derivative quantity.")
        return zip(pl_pz_list, x_list)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """前向传播"""
        raise NotImplementedError()

    @abstractmethod
    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        """
        计算本层所有参数的梯度（如果有）
        计算损失对本层输入的梯度
        :param pl_pz:
        :param x:
        :param index: 标记衍生变量顺序
        :return: 损失对本层输入的梯度
        """
        raise NotImplementedError()

    def backward(self, pl_pz_list: List[ndarray]) -> List[ndarray]:
        """
        反向传播
            计算所有参数的梯度
            传播完成后清除衍生变量
        :param pl_pz_list: 损失对本层输出的梯度
        :return: 损失对本层输入的梯度
        """
        pl_px_list = []
        index = 0
        for pl_pz, x in self.iterable_pl_pz_x(pl_pz_list):
            pl_px_list.append(self.calc_gradients(pl_pz, x, index))
            index += 1
        self.clear_derivations()
        return pl_px_list



class FunctionLayer(Layer):
    """
    无参数的Layer
    """
    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        """（反向传播）函数对输入的梯度"""
        return pl_pz * self.grad(x)

    @abstractmethod
    def grad(self, x: ndarray, **kwargs) -> ndarray:
        """（本质属性）函数对输入的梯度"""
        raise NotImplementedError()

    def __call__(self, x: ndarray):
        """弃用forward，改用fn"""
        return self.fn(x)

    @abstractmethod
    def fn(self, x: ndarray) -> ndarray:
        """弃用forward，改用fn"""
        raise NotImplementedError()

    def forward(self, x: ndarray) -> ndarray:
        """弃用forward，改用fn"""
        return self.fn(x)



class Activation(FunctionLayer):
    """激活函数"""

    @abstractmethod
    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        """计算梯度的梯度"""
        raise NotImplementedError()


class Affine(Activation):
    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def __str__(self) -> str:
        return f"{Affine.__name__}(slope={self.slope}, intercept={self.intercept})"

    @staticmethod
    def __fn__(x: ndarray, slope: float, intercept: float) -> ndarray:
        return slope * x + intercept

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x, self.slope, self.intercept)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return self.slope * np.ones_like(x)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.define_derivation("F")

    def __str__(self) -> str:
        return Sigmoid.__name__

    @staticmethod
    def __fn__(x: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-x))

    def fn(self, x: ndarray) -> ndarray:
        f = self.__fn__(x)
        if self.retain_derived:
            self.derivations['X'].append(x)
            self.derivations['F'].append(f)
        return f

    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        f = self.derivations['F'][index]
        return pl_pz * self.grad(x, f=f)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.__fn__(x)
        return f * (1 - f)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.__fn__(x)
        return f * (1 - f) * (1 - 2 * f)


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.define_derivation("F")

    def __str__(self) -> str:
        return Tanh.__name__

    @staticmethod
    def __fn__(x: ndarray) -> ndarray:
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def fn(self, x: ndarray) -> ndarray:
        f = self.__fn__(x)
        if self.retain_derived:
            self.derivations['X'].append(x)
            self.derivations['F'].append(f)
        return f

    def calc_gradients(self, pl_pz: ndarray, x: ndarray, index: int) -> ndarray:
        f = self.derivations['F'][index]
        return pl_pz * self.grad(x, f=f)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.__fn__(x)
        return 1 - f ** 2

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        f = kwargs['f'] if 'f' in kwargs else self.__fn__(x)
        return 2 * (f ** 3 - f)


class ReLU(Activation):
    def __str__(self) -> str:
        return ReLU.__name__

    @staticmethod
    def __fn__(x: ndarray) -> ndarray:
        # return np.clip(x, 0, np.inf)
        return np.maximum(x, 0)

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 1, 0)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        if isinstance(alpha, str):
            alpha = float(alpha)
        assert isinstance(alpha, float), "Unrecognized alpha type"
        self.alpha = alpha

    def __str__(self) -> str:
        return f"{LeakyReLU.__name__}(alpha={self.alpha})"

    @staticmethod
    def __fn__(x: ndarray, alpha: float) -> ndarray:
        return np.where(x > 0, x, x * alpha)

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x, self.alpha)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 1, self.alpha)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if isinstance(alpha, str):
            alpha = float(alpha)
        assert isinstance(alpha, float), "Unrecognized alpha type"
        self.alpha = alpha

    def __str__(self) -> str:
        return f"{ELU.__name__}(alpha={self.alpha})"

    @staticmethod
    def __fn__(x: ndarray, alpha: float) -> ndarray:
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x, self.alpha)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 1, self.alpha * np.exp(x))

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, 0, self.alpha * np.exp(x))


class Exponential(Activation):
    def __str__(self) -> str:
        return Exponential.__name__

    @staticmethod
    def __fn__(x: ndarray) -> ndarray:
        return np.exp(x)

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return self.__fn__(x)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return self.__fn__(x)


class SoftPlus(Activation):
    def __str__(self) -> str:
        return SoftPlus.__name__

    @staticmethod
    def __fn__(x: ndarray) -> ndarray:
        return np.log(1 + np.exp(x))

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        ex = kwargs['ex'] if 'ex' in kwargs else np.exp(x)
        return ex / (1 + ex)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        ex = kwargs['ex'] if 'ex' in kwargs else np.exp(x)
        return ex / ((1 + ex) ** 2)


class SELU(Activation):
    def __init__(self):
        super().__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)

    def __str__(self) -> str:
        return SELU.__name__

    @staticmethod
    def __fn__(x: ndarray, scale: float, elu: ELU) -> ndarray:
        return scale * elu.fn(x)

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x, self.scale, self.elu)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(
            x >= 0,
            np.ones_like(x) * self.scale,
            np.exp(x) * self.alpha * self.scale,
            )

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.where(x > 0, np.zeros_like(x), np.exp(x) * self.alpha * self.scale)


class HardSigmoid(Activation):
    def __str__(self) -> str:
        return HardSigmoid.__name__

    @staticmethod
    def __fn__(x: ndarray) -> ndarray:
        return np.clip((0.2 * x) + 0.5, 0.0, 1.0)

    def fn(self, x: ndarray) -> ndarray:
        if self.retain_derived:
            self.derivations['X'].append(x)
        return self.__fn__(x)

    def grad(self, x: ndarray, **kwargs) -> ndarray:
        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)

    def grad2(self, x: ndarray, **kwargs) -> ndarray:
        return np.zeros_like(x)


if __name__ == '__main__':
    def main():
        from matplotlib import pyplot as plt
        from matplotlib.axes import Axes

        def subplot(ax: Axes, x: ndarray, module: Activation):
            ax.plot(inputs, module(x), linestyle='-', label=r"$f(x)$")
            ax.plot(inputs, module.grad(inputs), linestyle='--', label=r"$\dfrac{df}{dx}$")
            ax.plot(inputs, module.grad2(inputs), linestyle=':', label=r"$\dfrac{d^2f}{dx^2}$")
            ax.set_xlim(-5, 5)
            # ax.set_ylim(-0.5, 1.5)
            ax.grid()
            ax.set_title(str(module))
            ax.legend()

        inputs = np.linspace(-5, 5).reshape(-1, 1)

        fig, axs = plt.subplots(1, 2, figsize=[12, 6], dpi=100)
        subplot(axs[0], inputs, Sigmoid())
        subplot(axs[1], inputs, Tanh())

        padding = [0.06, 0.03, 0.06, 0.03]
        plt.subplots_adjust(top=1-padding[0], right=1-padding[1], bottom=padding[2], left=padding[3])
        # plt.margins(0, 0)
        # plt.savefig("../../evaluate/激活函数对比.png")
        plt.show()
    main()
