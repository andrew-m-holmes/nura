import numpy as np
from typing import List


class Parameter:

    def __init__(self, data: float, grad=0) -> None:
        self.data = float(data)
        self.grad = grad
        self.backward = lambda: None

    def zero_grad(self):
        self.grad = 0

    def __add__(self, other: "Parameter"):
        param = Parameter(self.data + other.data)

        def backward():
            self.grad += param.grad
            other.grad += param.grad
            self.backward()
            other.backward()

        param.backward = backward
        return param

    def __mul__(self, other: "Parameter"):
        param = Parameter(self.data * other.data)

        def backward():
            self.grad += param.grad * other.data
            other.grad += param.grad * self.data
            self.backward()
            other.backward()

        param.backward = backward
        return param

    def __str__(self):
        return f"({self.data}, grad: {self.grad})"

    def __repr__(self) -> str:
        return f"({self.data}, grad: {self.grad})"


def forward(x: List[Parameter], w: Parameter):
    return [x[i] * w for i in range(len(x))]


def mse(y_true, y_pred):
    diff = [y_true[i] + Parameter(-1) * y_pred[i]
            for i in range(len(y_true))]
    squares = [diff[i] * diff[i] for i in range(len(y_true))]
    sums = sum(squares, Parameter(0))
    error = Parameter(1 / len(y_true)) * sums
    return error


def main():

    # setup data
    slope = 3.0
    x = [Parameter(i * 0.1) for i in range(-100, 100)]
    y = [Parameter(slope) * x[i] + Parameter(4)
         for i in range(len(x))]

    # f(x) = m * x + 4

    # setup learnable parameter
    w = np.random.rand(1)
    w = Parameter(w.item())

    # setup training
    iters = 100
    learning_rate = 1e-3
    net_loss = 0

    print(f"w before: {w}")

    # runing stochastic gradient descent
    for iter in range(iters):

        # predict and backprop
        pred = forward(x, w)
        loss = mse(y, pred)

        loss.grad = 1.0
        loss.backward()
        w.data = w.data - w.grad * learning_rate
        w.zero_grad()
        net_loss += loss.data

    avg_loss = net_loss / iters

    print(f"w after: {w}")
    print(f"average mean squared error: {avg_loss}")

    # dy/dx = m

    print(w * Parameter(10))


if __name__ == "__main__":
    main()
