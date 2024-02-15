import numpy as np


class Primal:

    def __init__(self, array, adjoint=0):
        self.array = array
        self.adjoint = adjoint
        self.backward = None

    def zero(self):
        self.adjoint = 0

    def __add__(self, other):
        array = self.array + other.array
        primal = Primal(array)

        def backward(adjoint):
            primal.adjoint += adjoint
            self.adjoint += adjoint * 1
            other.adjoint += adjoint * 1

        primal.backward = backward
        return primal

    def __sub__(self, other):
        array = self.array - other.array
        primal = Primal(array)

        def backward(adjoint):
            primal.adjoint += adjoint
            self.adjoint += adjoint * 1
            other.adjoint += adjoint * -1

        primal.backward = backward
        return primal

    def __mul__(self, other):
        array = self.array * other.array
        primal = Primal(array)

        def backward(adjoint):
            primal.adjoint += adjoint
            self.adjoint += adjoint * other.array
            other.adjoint += adjoint * self.array

        primal.backward = backward
        return primal

    def __truediv__(self, other):
        array = self.array / other.array
        primal = Primal(array)

        def backward(adjoint):
            primal.adjoint += adjoint
            self.adjoint += adjoint / other.array
            other.adjoint += adjoint * (-self.array / other.array**2)

        primal.backward = backward
        return primal

    def __matmul__(self, other):
        array = self.array @ other.array
        primal = Primal(array)

        def backward(adjoint):
            primal.adjoint += adjoint
            self.adjoint += adjoint @ other.array.T
            other.adjoint += self.array.T @ adjoint

        primal.backward = backward
        return primal

    def __repr__(self):
        return f"primal: {self.array}, adjoint: {self.adjoint}"


def main():
    np.set_printoptions(precision=4)
    a = Primal(np.array(5.0))
    b = Primal(np.array(15.0))

    c = a * b 
    d = c * a
    d.backward(np.array(1.))
    print(f"a * b * a = {d}")
    print(f"{a = }, {b = }, {c = }")


if __name__ == "__main__":

    main()
