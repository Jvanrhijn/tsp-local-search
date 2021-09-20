import math
import numpy as np
import scipy.special as spec
import scipy.integrate as integ
import matplotlib.pyplot as plt



def func1(x):
    return 1/np.log(x)**(np.log(np.log(x)))

def func2(x):
    return 1/np.log(x)**6


xs = np.arange(2, 10000_001)

f1 = func1(xs)
f1int = integ.cumtrapz(f1, x=xs, initial=0)

f2 = func2(xs)
f2int = integ.cumtrapz(f2, x=xs, initial=0)


plt.figure()
plt.loglog(xs, f1int)
plt.loglog(xs, 10000*f2int)
plt.show()
