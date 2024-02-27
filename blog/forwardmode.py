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

    def mul_add(a, b, c):
        return a * b + c * a

    def div_sub(a, b, c):
        return a / b - c

    a, b, c = Primal(25.0, 1.0), Primal(4.0, 0.0), Primal(-5.0, 0.0)
    print(f"{a = }, {b = }, {c = }")
    print(f"{mul_add(a, b, c) = }")
    a.tangent, b.tangent, c.tangent = 0.0, 1.0, 0.0
    print(f"{div_sub(a, b, c) = }")


if __name__ == "__main__":
    main()
