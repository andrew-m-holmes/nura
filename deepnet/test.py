import deepnet.nn.functional as f
from deepnet.tensor import Tensor


def main():
    a = Tensor([3], use_grad=True)
    b = Tensor([5], use_grad=True)

    c = a + b
    d = c * a
    print(d)
    print(d.grad_fn.next_functions)
    d.backward()
    print(a.grad, b.grad, c.grad)


if __name__ == "__main__":
    main()
