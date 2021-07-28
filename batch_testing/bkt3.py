import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def sens_burns(n, k, sp_i, sens_i, a):
    if k > 0:
        return 1 - sp_i + (sens_i + sp_i - 1) * (k/n) ** a 
    else:
        return sens_i


def sens_hwang(n, k, p, d, sens_i):
    if k > 0:
        return p/(1 - (1-p) ** (d * n))
    else:
        return sens_i


def linear_scan(n_max, sens_func, p, sp, **kwargs):
    def obj(sens_func, n, p, sp, **kwargs):
        temp = sum([(sens_func(n, k, **kwargs) * math.comb(n, k) * (p/(1-p)) ** k) for k in range(1, n+1)])
        return ((1-p) ** n * (1/(1-p) ** n + n - n*sp + n * temp)) / n

    temp = [obj(sens_func, n, p, sp, **kwargs) for n in range(1, n_max + 1)]
    return temp.index(min(temp)) + 1


kweys = {'sens_i': 0.95,  'sens_func': sens_burns, 'sp_i': 0.99}
print(linear_scan(10, p = 0.05, sp = 0.95, a= 0.87,**kweys))
print(sens_burns(32, 32, 0.99, 0.95, 0.37))

a = [sens_burns(32, k, 0.99, 0.95, 0.33) for k in range(1, 33)]
b = [sens_hwang(32, k, 0.05, 0.01, 0.95) for k in range(1, 33)]


plt.plot(a)
plt.plot(b)
plt.show()