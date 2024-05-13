import nura
import nura.nn as nn
import nura.autograd.functional as f


def main():

    a = nura.tensor(15).float().attached()
    b = a * 3
    f.backward(b, 1.0)
    print(a.grad)


if __name__ == "__main__":
    main()
