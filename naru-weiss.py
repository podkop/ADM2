from scipy.optimize import shgo
import numpy as np

nvar=2 # nr. of variables
nfun=4 # nr. of functions

## objective functions
def f1(x):
    return -4.07-2.27*x[0]
def f2(x):
    return -2.6-0.03*x[0]-0.02*x[1]-0.01/(1.39-x[0]**2)-0.3/(1.39-x[1]**2)
def f3(x):
    return -8.21+0.71/(1.09-x[0]**2)
def f4(x):
    return -0.96+0.96/(1.09-x[1]**2)
fvect=[f1,f2,f3,f4]
def f(x): # the vector objective function
    return [f1(x),f2(x),f3(x),f4(x)]
def sumf(x,*args):
    return args[0]*sum(f(x))

## ideal, nadir
ideal=np.array([-6.34,-3.44,-7.5,0])
nadir=([-4.07,-2.83,-0.32,9.71])

## weights for Chebyshev
w=1/(nadir-ideal)

## bounds
lb=0.3
ub=1
bnd=[(lb,ub) for i in range(nvar)]


def solve(refp):
    