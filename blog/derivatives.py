import matplotlib.pyplot as plt
import numpy as np


def main():

    def f(x):
        return np.cos(x)

    def df(x):
        return -np.sin(x)

    def tangent_line(x, x0, y0, slope):
        return slope * (x - x0) + y0

    x = np.linspace(0, 4, 150)
    h = 0.5
    x0 = 2.0
    y0 = f(x0)
    true_dx = df(x0)
    forward_dx = (f(x0 + h) - f(x0)) / h
    backward_dx = (f(x0) - f(x0 - h)) / h
    central_dx = (f(x0 + h) - f(x0 - h)) / (2 * h)

    y = f(x)
    true_tangent = tangent_line(x, x0, y0, true_dx)
    forward_tangent = tangent_line(x, x0, y0, forward_dx)
    backward_tangent = tangent_line(x, x0, y0, backward_dx)
    central_tangent = tangent_line(x, x0, y0, central_dx)

    plt.style.use("dark_background")
    plt.plot(x, y, label="cos(x)", color="dodgerblue")
    plt.plot(x, true_tangent, label=f"Tangent", color="red")
    plt.plot(x, forward_tangent, label="Forward Difference", color="aquamarine")
    plt.plot(x, backward_tangent, label="Backward Difference", color="lightcoral")
    plt.plot(x, central_tangent, label="Central Difference", color="blueviolet")
    plt.legend()

    plt.axhline(0, color="white")
    plt.axvline(0, color="white")

    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.show()
    plt.savefig("./images/derivatives.png", dpi=300)


if __name__ == "__main__":
    main()
