import numpy as np

np.set_printoptions(precision=4)


class Primal:

    def __init__(self, data, tangent):
        self.data = data
        self.tangent = tangent

    def __add__(self, other):
        data = self.data + other.data
        tangent = self.tangent + other.tangent
        return Primal(data, tangent)

    def __sub__(self, other):
        data = self.data - other.data
        tangent = self.tangent - other.tangent
        return Primal(data, tangent)

    def __mul__(self, other):
        data = self.data * other.data
        tangent = self.tangent * other.data + other.tangent * self.data
        return Primal(data, tangent)

    def __truediv__(self, other):
        data = self.data / other.data
        tangent = (self.tangent / other.data) + (
            -self.data / other.data**2
        ) * other.tangent
        return Primal(data, tangent)

    def __repr__(self):
        return f"primal: {self.data}, tangent: {self.tangent}"


def main():

    np.set_printoptions(precision=4)

    def mul_add(a, b, c):
        return a * b + c

    def div_sub(a, b, c):
        return b / c - a

    a = Primal(np.array(3.0), np.array(1.0))
    b = Primal(np.array(4.0), np.array(0.0))
    c = Primal(np.array(1.0), np.array(2.0))

    print(f"{mul_add(a, b, c) = }")
    print(f"{div_sub(a, b, c) = }")

if __name__ == "__main__":

    main()
