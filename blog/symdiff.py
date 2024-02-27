from sympy import *

x = symbols("x")
fog = 4 * (cos(x) + 2 * x - exp(x)) ** 2
df_dx = diff(fog, x)
print(df_dx)