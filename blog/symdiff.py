from sympy import symbols, cos, exp, diff


def main():
    x = symbols("x")
    fog = 4 * (cos(x) + 2 * x - exp(x)) ** 2
    dfdx = diff(fog, x)
    print(dfdx)

    def f(x):
        if x > 2:
            return x * 2 + 5
        return x / 2 + 5

    dfdx = diff(f)
    print(dfdx)


if __name__ == "__main__":
    main()
