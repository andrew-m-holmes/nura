import numpy as np 
import deepnet

def main():

    data = np.random.randint(0, 10, (2, 2))
    a = deepnet.tensor(data, diff=True).float()
    b = deepnet.tensor(data * -5., diff=True).float()

if __name__ == "__main__":
    main()
