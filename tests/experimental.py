import numpy as np 
import deepnet

def main():

    data = np.random.randint(0, 10, (2, 2))
    a = deepnet.tensor(data)
    print(a)
    b = a.float()
    print(b is a)
    print(b)
    c = b.dual()
    print(c)
    print(c.unpack())

if __name__ == "__main__":
    main()
