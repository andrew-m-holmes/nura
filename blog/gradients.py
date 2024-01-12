import matplotlib.pyplot as plt
import numpy as np

def main():

    def f(x):
        return np.power(x, 2) 

    def df(x):
        return 2 * x

    def tangent_line(x, x0, y0, slope):
        return slope * (x - x0) + y0

    x = np.linspace(0, 4, 100)
    h = 0.5
    x0 = 2  
    y0 = f(x0)
    true_grad = df(x0)
    approx_grad = (f(x0 + h) - f(x0)) / h
    
    print(true_grad)
    print(approx_grad)

    y = f(x)
    true_tangent = tangent_line(x, x0, y0, true_grad)
    approx_tangent = tangent_line(x, x0, y0, approx_grad)

    plt.style.use("dark_background")
    plt.plot(x, y, label="f(x)")
    plt.plot(x, true_tangent, label="True Gradient", color="red")
    plt.plot(x, approx_tangent, label="Central Difference Gradient", color="purple")
    plt.legend()

    plt.axhline(0, color="white")  
    plt.axvline(0, color="white")  

    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))

    plt.show()


if __name__ == "__main__":
    main()