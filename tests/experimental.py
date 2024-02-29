import neuro
import neuro.nn as nn
import numpy as np


def main():

    linear = nn.Linear(3, 5, bias=True)
    inpt = neuro.rand((2, 3)).float()
    out = neuro.sum(linear(inpt))
    out.backward()
    assert linear.weight.tensor.grad is not None 
    assert linear.bias.tensor.grad is not None 

    linear = nn.Linear(3, 5, bias=False)
    out = neuro.sum(linear(inpt))
    out.backward()
    assert linear.weight.tensor.grad is not None 
    assert linear.bias is None 

if __name__ == "__main__":
    main()
