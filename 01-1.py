import numpy as np
import matplotlib.pyplot as plt

true_b = 1
true_w = 2

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)
N = 100
lr = 0.1
# 定义周期数
n_epochs = 1000

X = np.random.rand(N, 1)
y = true_w * X + true_b + (0.1 * np.random.randn(N, 1))

idx = np.arange(N)
np.random.shuffle(idx)

X_train, y_train = X[:int(N * 0.8)], y[:int(N * 0.8)]
X_val, y_val = X[int(N * 0.8):], y[int(N * 0.8):]

losses = []
for epoch in range(n_epochs):
    yhat = X_train * w + b
    error = yhat - y_train
    loss = (error ** 2).mean()

    b_grad = 2 * error.mean()
    w_grad = 2 * (error * X_train).mean()

    b -= b_grad * lr
    w -= w_grad * lr
    losses.append(loss)

print(b, w)
plt.plot(losses)
plt.show()
