import numpy as np 
np.set_printoptions(precision=4)

class Primal:

    def __init__(self, array, tangent):
        self.array = array
        self.tangent = tangent

    def __add__(self, other):
        array = self.array + other.array
        tangent = self.tangent + other.tangent
        return Primal(array, tangent)

    def __sub__(self, other):
        array = self.array - other.array
        tangent = self.tangent - other.tangent
        return Primal(array, tangent)

    def __mul__(self, other):
        array = self.array * other.array
        tangent = self.tangent * other.array + other.tangent * self.array
        return Primal(array, tangent)

    def __truediv__(self, other):
        array = self.array / other.array
        tangent = (self.tangent / other.array) + (-self.array / other.array ** 2) * other.tangent 
        return Primal(array, tangent)

    def __matmul__(self, other):
        array = self.array @ other.array
        tangent = self.tangent @ other.array + self.array @ other.tangent
        return Primal(array, tangent)

    def __repr__(self):
        return f"primal: {self.array}, tangent: {self.tangent}"

def main():

    np.set_printoptions(precision=4)

    a = Primal(np.array(5.), np.array(1.))
    b = Primal(np.array(15.), np.array(0))
    print(f"{a = } {b = }")
    print(f"{a + b = }")
    print(f"{a - b = }")
    print(f"{a * b = }")
    print(f"{b / a = }")

    m1 = Primal(np.random.rand(3, 2), np.ones((3, 2)))
    m2 = Primal(np.random.rand(2, 1), np.zeros((2, 1)))
    print(f"{m1 @ m2 = }")

if __name__ == "__main__":

    main()



