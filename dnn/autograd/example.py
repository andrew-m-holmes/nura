import numpy as np


class Parameter:

    def __init__(self, data: float, grad=0) -> None:
        self.data = float(data)
        self.grad = grad
        self.backward = lambda: None

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
        return f"(data: {self.data}, grad: {self.grad})"


def main():

    a = Parameter(3)
    b = Parameter(2)
    print(f"a: {a}, b: {b}")

    c = a * b
    print(f"a: {a}, b: {b}, c: {c}")

    d = Parameter(5)
    print(f"d: {d}")

    e = c * d
    print(f"e: {e}")

    e.grad = 1.0
    e.backward()
    print(f"a: {a}, b: {b}, c: {c}, d: {e}")


if __name__ == "__main__":
    main()
