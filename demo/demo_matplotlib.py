import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes, SubplotBase
matplotlib.use('TkAgg')


if __name__ == '__main__':
    fig: Figure = plt.figure()
    fig.suptitle("Title")
    ax11: Axes = fig.add_subplot(2, 2, 1)
    ax11.plot(range(10), label="11", linestyle=":")
    ax11.legend()
    ax11.grid()
    ax11.set_xlabel("X")
    ax11.set_ylabel("Y")
    ax11.set_title("11")
    #
    ax12: Axes = fig.add_subplot(2, 2, 2)
    ax12.scatter(range(20), np.random.randn(20), label="12", marker="*")
    ax12.legend()
    ax12.grid()
    ax12.set_xlabel("X")
    ax12.set_ylabel("Y")
    ax12.set_title("12")
    #
    plt.show()
