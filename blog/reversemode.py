class Variable:

    def __init__(self, primal, adjoint=0.0):
        self.primal = primal
        self.adjoint = adjoint

    def backward(self, adjoint):
        self.adjoint += adjoint

    def __add__(self, other):
        variable = Variable(self.primal + other.primal)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * 1.0
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __sub__(self, other):
        variable = Variable(self.primal - other.primal)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * -1.0
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __mul__(self, other):
        variable = Variable(self.primal * other.primal)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * other.primal
            other_adjoint = adjoint * self.primal
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __truediv__(self, other):
        variable = Variable(self.primal / other.primal)

        def backward(adjoint):
            variable.adjoint += adjoint
            self_adjoint = adjoint * (1.0 / other.primal)
            other_adjoint = adjoint * (-1.0 * self.primal / other.primal**2)
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __repr__(self) -> str:
        return f"primal: {self.primal}, adjoint: {self.adjoint}"


def main():
    def mul_add(a, b, c):
        return a * b + c * a

    def div_sub(a, b, c):
        return a / b - c

    a, b, c = Variable(25.0, 1.0), Variable(4.0, 0.0), Variable(-5.0, 0.0)

    print(f"{a = }, {b = }, {c = }")
    d = mul_add(a, b, c)
    d.backward(1.0)
    print(f"{d = }")
    print(f"{a.adjoint = }, {b.adjoint = }, {c.adjoint = }")

    a.adjoint, b.adjoint, c.adjoint = 0.0, 0.0, 0.0
    e = div_sub(a, b, c)
    e.backward(1.0)
    print(f"{e = }")
    print(f"{a.adjoint = }, {b.adjoint = }, {c.adjoint = }")


if __name__ == "__main__":
    main()
