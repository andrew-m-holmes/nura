import neuro
import neuro.nn as nn


def main():


    linear = nn.Linear(4, 5, bias=True)
    x = neuro.rand((1, 4))
    y = linear(x).sum()
    y.backward()


if __name__ == "__main__":
    main()
