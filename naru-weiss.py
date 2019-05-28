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
    return np.array([f1(x),f2(x),f3(x),f4(x)])

## ideal, nadir
ideal=np.array([-6.34,-3.44,-7.5,0])
nadir=np.array([-4.07,-2.83,-0.32,9.71])

## weights for Chebyshev
w=1/(nadir-ideal)

## bounds
lb=0.3
ub=1
bnd=[(lb,ub) for i in range(nvar)]+[(None,None)]


### PROBLEM FORMULATION
#   decision vector xt = [x1_1,...,x_n,t]

## ASF objective: t + augmentation term 
#  *args = ( ref.point(list), rho(float) )
def rhosum_f(xt,*args):
    return (
           xt[-1]+ # t
           args[1] * sum( w*(f(xt[:-1])-args[0]) ) # augm. term
           )
## l.h.s. fcuntions for ASF constraints: t >= w_i(f_i(x)-ref.point_i), i=1..n
t_constr=[ # *args = (ref.point_i)
        lambda xt,*args: -w[0]*fvect[0](xt[:-1])+xt[-1]+w[0]*args[0],
        lambda xt,*args: -w[1]*fvect[1](xt[:-1])+xt[-1]+w[1]*args[0],
        lambda xt,*args: -w[2]*fvect[2](xt[:-1])+xt[-1]+w[2]*args[0],
        lambda xt,*args: -w[3]*fvect[3](xt[:-1])+xt[-1]+w[3]*args[0]        
        ]

### PROBLEM SOLVING

def solve_ref(refp,sampl_m='simplicial',itern=1,npoints=100):
    sol = shgo(
            rhosum_f, #obj. function
            bounds=bnd, # variable bounds
            args=(np.array(refp),10^-10), # parameters for obj. func.
            constraints=[ # constraints
             {
              "type": "ineq",
              "fun": t_constr[i],
              "args": (refp[i]+2/w[i],)
              }                    
            for i in range(nfun)
            ],
            sampling_method=sampl_m,
            iters=itern,
            n=npoints
           )
    constr=[
            t_constr[i](sol["x"],refp[i])
            for i in range(nfun)
            ]
    if min(constr)>10**-6:
        print("Wrong constraints for ",refp,": ",constr)
    return {"message": sol["message"],
            "x": sol["x"],
            "fun": sol["fun"],
            "nfev": sol["nfev"],
            "nlfev":sol["nlfev"],
            "constr":constr}

for i in range(100):
    rfp=[ideal[i]+np.random.random_sample()*(nadir[i]-ideal[i])
            for i in range(nfun)]
    print(solve_ref(rfp,itern=5)["x"])
# -4.07,-2.82,-3,4 -> -0.3265958717892198
# -5.54,-3.12,-3.33,3.78 -> no