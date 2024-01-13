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
    h = 0.7
    x0 = 2.0 
    y0 = f(x0)
    true_grad = df(x0)
    forward_grad = (f(x0 + h) - f(x0)) / h
    central_grad = (f(x0 + h) - f(x0 - h)) / (2 * h)
    

    y = f(x)
    true_tangent = tangent_line(x, x0, y0, true_grad)
    forward_tangent = tangent_line(x, x0, y0, forward_grad)
    central_tangent = tangent_line(x, x0, y0, central_grad)

    plt.style.use("dark_background")
    plt.plot(x, y, label="cos(x)", color="dodgerblue")
    plt.plot(x, true_tangent, label=f"Gradient", color="red")
    plt.plot(x, forward_tangent, label="Forward Difference", color="aquamarine")
    plt.plot(x, central_tangent, label="Central Difference", color="blueviolet")
    plt.legend()

    plt.axhline(0, color="white")  
    plt.axvline(0, color="white")  

    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))

    plt.show()


if __name__ == "__main__":
    main()