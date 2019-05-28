from scipy.optimize import shgo
import numpy as np

nvar=2 # nr. of variables
nfun=3 # nr. of functions

## objective functions
def psi(x):
    return x[0]*x[0]+x[1]*x[1]
def phi(x):
    return psi(x)-np.exp(-50*psi(x))

def f1(x):
    return phi(x)
def f2(x):
    return phi([x[0],x[1]-1])
def f3(x):
    return phi([x[0]-1,x[1]])

fvect=[f1,f2,f3]
def f(x): # the vector objective function
    return np.array([f1(x),f2(x),f3(x)])

## ideal, nadir
ideal=np.array([-1,-1,-1])
nadir=np.array([1,2,2])

## weights for Chebyshev
w=1/(nadir-ideal)

## bounds
bnd=[(-10,10) for i in range(nvar+1)]


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
        ]

### PROBLEM SOLVING

def solve_ref(refp,sampl_m='simplicial',itern=1,npoints=100):
    sol = shgo(
            rhosum_f, #obj. function
            bounds=bnd, # variable bounds
            args=(np.array(refp),10**-6), # parameters for obj. func.
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
            n=npoints,
            options={"minimize_every_iter":True,"local_iter":False}
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
if __name__=="__main__":
    for i in range(100):
        rfp=[ideal[i]+np.random.random_sample()*(nadir[i]-ideal[i])
                for i in range(nfun)]
        print(solve_ref(rfp,itern=5)["x"][:-1],"\n")
    #sol=solve_ref([-1,-1,-1],itern=6)
    #print(sol)

