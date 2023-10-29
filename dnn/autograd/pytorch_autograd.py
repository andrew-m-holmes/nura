import torch
import torch.nn as nn


def predict(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    z = torch.matmul(x, w) + b
    a = torch.sigmoid(z)
    return a


def main():

    n_features, m = 100, 10000
    x = torch.randn(m, n_features)
    y = torch.randint(0, 2, (m, 1)).float()

    w = torch.randn(n_features, 1)
    w.requires_grad_()
    b = torch.zeros(1, requires_grad=True)

    iters = 1000
    learning_rate = 1e-1
    print_every = 250

    for i in range(iters):
        out = predict(x, w, b)
        loss = nn.functional.binary_cross_entropy(out, y)
        loss.backward()

        # grads acucmulated so just need to update params w/ their grads
        with torch.no_grad():
            w.data -= w.grad * learning_rate  # Removed division by m
            b.data -= b.grad * learning_rate

        if (i + 1) % print_every == 0 and i != 1:
            # print(w.grad.tolist())
            # print(b.grad.tolist())
            print(f"iter: {i + 1}: {loss:.4f}")

        # zero grads to prevent old grads being apart of backprop step for next iter
        w.grad.zero_()
        b.grad.zero_()


if __name__ == "__main__":
    main()
