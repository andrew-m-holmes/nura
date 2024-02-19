import torch


class Primal:

    def __init__(self, data, adjoint=0.0):
        self.data = data
        self.adjoint = adjoint

    def backward(self, adjoint):
        self.adjoint += adjoint

    def __add__(self, other):
        primal = Primal(self.data + other.data)

        def backward(adjoint):
            primal.adjoint += adjoint
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * 1.0
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        primal.backward = backward
        return primal

    def __sub__(self, other):
        primal = Primal(self.data - other.data)

        def backward(adjoint):
            primal.adjoint += adjoint
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * -1.0
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        primal.backward = backward
        return primal

    def __mul__(self, other):
        primal = Primal(self.data * other.data)

        def backward(adjoint):
            primal.adjoint += adjoint
            self_adjoint = adjoint * other.data
            other_adjoint = adjoint * self.data
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        primal.backward = backward
        return primal

    def __truediv__(self, other):
        primal = Primal(self.data / other.data)

        def backward(adjoint):
            primal.adjoint += adjoint
            self_adjoint = adjoint * (1.0 / other.data)
            other_adjoint = adjoint * (-1.0 * self.data / other.data**2)
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        primal.backward = backward
        return primal

    def __repr__(self) -> str:
        return f"primal: {self.data}, adjoint: {self.adjoint}"


def main():
    def mul_add(a, b, c):
        return a * b + c * a

    def div_sub(a, b, c):
        return a / b - c

    a, b, c = Primal(9.0), Primal(3.0), Primal(-5.0)
    print(f"{a = }, {b = }, {c = }")

    d = mul_add(a, b, c)
    d.backward(1.0)
    print(f"{d = }")
    print(f"{a.adjoint = }, {b.adjoint = }, {c.adjoint = }")

    a.adjoint, b.adjoint, c.adjoint = 0.0, 0.0, 0.0

    e = div_sub(a, b, c)
    e.backward(3.0)
    print(f"{e = }")
    print(f"{a.adjoint = }, {b.adjoint = }, {c.adjoint = }")


if __name__ == "__main__":
    main()
