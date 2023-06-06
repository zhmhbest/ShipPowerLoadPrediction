import numpy as np
from sklearn.svm import SVR  # 回归

if __name__ == '__main__':
    x_train = np.random.randn(256, 7)
    y_train = np.random.randn(256, 1).reshape(-1)

    model = SVR()
    model.fit(x_train, y_train)
    # model.predict()
