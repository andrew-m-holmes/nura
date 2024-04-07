import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np

def main():
    
    x = nura.randn(2, 7, 3, usegrad=True)
    gamma = nura.randn(7, 3, usegrad=True)
    beta = nura.randn(7, 3, usegrad=True)
    z = f.layernorm(x, gamma, beta, bias=True, dim=(-2, -1))
    z.backward(nura.oneslike(z))

if __name__ == "__main__":
    main()
