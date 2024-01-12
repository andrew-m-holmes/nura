import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.linspace(0, 100, 50)
    y = np.power(x, 2)
    plt.style.use("dark_background")
    plt.plot(x, y, color="red")

    plt.axhline(0, color='white')  
    plt.axvline(0, color='white')  

    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))

    plt.show()
    plt.show()


if __name__ == "__main__":
    main()