import torch
import torch.nn as nn
import numpy as np


def predict(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    z = x * w + b
    return z


def main():

    x = torch.arange(-10, 10).unsqueeze(0).float()
    m = 3
    a = 5
    y = m * x + a  # func

    # params
    w = torch.rand((1, 1), requires_grad=True)
    b = torch.zeros((1,), requires_grad=True)

    learning_rate = 1e-2
    iters = 1000

    for iter in range(iters):

        out = predict(x, w, b)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()

        with torch.no_grad():
            w.data = w.data - w.grad * learning_rate
            b.data = b.data - b.grad * learning_rate

        w.grad.zero_()
        b.grad.zero_()

    assert np.isclose(
        w.item(), m, rtol=1e-3), "incorrect weight learned"
    assert np.isclose(
        b.item(), a, rtol=1e-3), "incorret bias learned"


if __name__ == "__main__":
    main()
