class Variable:

    def __init__(self, primal, tangent):
        self.primal = primal
        self.tangent = tangent

    def __add__(self, other):
        primal = self.primal + other.primal
        tangent = self.tangent + other.tangent
        return Variable(primal, tangent)

    def __sub__(self, other):
        primal = self.primal - other.primal
        tangent = self.tangent - other.tangent
        return Variable(primal, tangent)

    def __mul__(self, other):
        primal = self.primal * other.primal
        tangent = self.tangent * other.primal + other.tangent * self.primal
        return Variable(primal, tangent)

    def __truediv__(self, other):
        primal = self.primal / other.primal
        tangent = (self.tangent / other.primal) + (
            -self.primal / other.primal**2
        ) * other.tangent
        return Variable(primal, tangent)

    def __repr__(self):
        return f"primal: {self.primal}, tangent: {self.tangent}"


def main():

    def mul_add(a, b, c):
        return a * b + c * a

    def div_sub(a, b, c):
        return a / b - c

    a, b, c = Variable(25.0, 1.0), Variable(4.0, 0.0), Variable(-5.0, 0.0)
    print(f"{a = }, {b = }, {c = }")
    print(f"{mul_add(a, b, c) = }")
    a.tangent, b.tangent, c.tangent = 0.0, 1.0, 0.0
    print(f"{div_sub(a, b, c) = }")


if __name__ == "__main__":
    main()
