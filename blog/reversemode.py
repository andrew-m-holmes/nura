import numpy as np


class Primal:

    def __init__(self, data, adjoint=0):
        self.data = data
        self.adjoint = adjoint
        self.backward = lambda adjoint: None

    def __add__(self, other):
        data = self.data + other.data
        primal = Primal(data)

        def backward(adjoint):
            self.backward(adjoint)
            other.backward(adjoint)
            self.adjoint += adjoint
            other.adjoint += adjoint

        primal.backward = backward
        return primal

    def __sub__(self, other):
        data = self.data - other.data
        primal = Primal(data)

        def backward(adjoint):
            self.backward(adjoint)
            other.backward(-adjoint)
            self.adjoint += adjoint
            other.adjoint += -adjoint

        primal.backward = backward
        return primal

    def __mul__(self, other):
        data = self.data * other.data
        primal = Primal(data)

        def backward(adjoint):
            self_adjoint = adjoint * other.data
            other_adjoint = adjoint * self.data
            self.backward(self_adjoint)
            other.backward(other_adjoint)
            self.adjoint += self_adjoint
            other.adjoint += other_adjoint

        primal.backward = backward
        return primal

    def __truediv__(self, other):
        data = self.data / other.data
        primal = Primal(data)

        def backward(adjoint):
            self_adjoint = adjoint / other.data
            other_adjoint = adjoint * (-self.data / other.data**2)
            self.backward(self_adjoint)
            other.backward(other_adjoint)
            self.adjoint += self_adjoint
            other.adjoint += other_adjoint

        primal.backward = backward
        return primal

    def __repr__(self):
        return f"primal: {self.data}, adjoint: {self.adjoint}"


def main():

    a = Primal(np.full((2, 2), 9.0)) 
    b = Primal(np.full((2, 2), 3.0))
    c = Primal(np.full((2, 2), 2.0))

    def mul_add(a, b, c):
        return a * b + c

    def div_sub(a, b, c):
        return a / b - c

    output = mul_add(a, b, c)
    output.backward(np.ones_like(output))
    print(f"mul_add mul_add(a, b, c) = \n{output}")
    print(a, b, c, sep="\n")

    output = div_sub(a, b, c)
    output.backward(np.ones_like(output))
    print(f"div_sub(a, b, c) = \n{output}")
    print(a, b, c, sep="\n")

if __name__ == "__main__":
    main()
