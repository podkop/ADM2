from scipy.optimize import shgo
import numpy as np

## non-dominated sorting
#import pygmo as pg

#outputting
from matplotlib import pyplot as plt


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

## basic weights for Chebyshev
w0=1/(nadir-ideal)

## bounds
bnd=[(0,1) for i in range(nvar)]+[(-10,10)]


### PROBLEM FORMULATION
#   decision vector xt = [x1_1,...,x_n,t]

## ASF objective: t + augmentation term 
#  *args = ( ref.point(list), rho(float), weights(list))
def rhosum_f(xt,*args):
    return (
           xt[-1]#+ # t
           #args[1] * sum( args[2]*(f(xt[:-1])-args[0]) ) # augm. term
           )
## l.h.s. fcuntions for ASF constraints: t >= w_i(f_i(x)-ref.point_i), i=1..n
t_constr=[ # *args = (ref.point_i,weight_i)
        lambda xt,*args: 
            -args[1]*fvect[0](xt[:-1])+xt[-1]+args[1]*args[0] - 10**-8*sum(f(xt[:-1])*args[1]),
        lambda xt,*args: 
            -args[1]*fvect[1](xt[:-1])+xt[-1]+args[1]*args[0] - 10**-8*sum(f(xt[:-1])),
        lambda xt,*args: 
            -args[1]*fvect[2](xt[:-1])+xt[-1]+args[1]*args[0] - 10**-8*sum(f(xt[:-1])),
        lambda xt: -fvect[0](xt[:-1]) + 1 # keep artificial upper bound    
        ]

### PROBLEM SOLVING

def solve_ref(refpoint,w,sampl_m='simplicial',itern=1,npoints=100):
    # shifting the ref. point to exceed nadir+(nadir-ideal)
    tadd=max(0,max(w*(4*nadir-3*ideal-refpoint)))
    refp=refpoint+tadd/w
    # calling the solver
    sol = shgo(
            rhosum_f, #obj. function
            bounds=bnd, # variable bounds
            args=(np.array(refp),10**-6,w), # parameters for obj. func.
            constraints=[ # constraints
                 {
                  "type": "ineq",
                  "fun": t_constr[i],
                  "args": (refp[i],w[i])
                  }                    
                for i in range(nfun)
                ]+[{
                    "type": "ineq",
                    "fun": t_constr[3]
                        }],
            sampling_method=sampl_m,
            iters=itern,
            n=npoints,
            options={"minimize_every_iter":True,"local_iter":False}
           )
    # are ASF constraints tight?
    constr=[
            t_constr[i](sol["x"],refp[i],w[i])
            for i in range(nfun)
            ]
    if min(constr)>10**-6:
        print("Wrong constraints for ",refp,": ",constr)
    # return
    return {"message": sol["message"],
            "x": sol["x"],
            "fun": sol["fun"],
            "nfev": sol["nfev"],
            "nlfev":sol["nlfev"],
            "constr":constr}
if __name__=="__main__":
    pass
### generating and saving random reference points
#    ref_l=np.array([
#            [ideal[i]+np.random.random_sample()*(nadir[i]-ideal[i])
#                for i in range(nfun)]
#            for i in range(500)])
#    np.savetxt("reflist.txt",ref_l)
### loading same saved reference points
#    ref_l=np.genfromtxt("reflist.txt")
### collecting solution results    
#    x_l=[]
#    y_l=[]
#    for i in range(500):
#        res=solve_ref(ref_l[i],w0,itern=5)
#        x_l.append(res["x"])
#        y_l.append(f(res["x"]))
###    saving solution results
#    np.savetxt("aug_add_10-6_x.txt",x_l)
#    np.savetxt("aug_add_10-6_y.txt",y_l)


    ### Testing results of different ASF formulations on random points
    asfnames=["aug_10-6", # proper augmentation via constraints
              "noaugm", # no augmentation
              "aug_add_10-6" # augmented by adding term to the obj. function
              ]    
    xx=[np.genfromtxt(s+"_x.txt") for s in asfnames]
    yy=[np.genfromtxt(s+"_y.txt") for s in asfnames]
    
    ### checking non-domination
    #np.set_printoptions(precision=5)
    #for i,s in enumerate(asfnames):
    #    print("\n*******\n"+s)
    #    for i1,y1 in enumerate(yy[i]):
    #        for i2,y2 in enumerate(yy[i]):
    #            if max(y1-y2)<=10**-6 and min(y1-y2)<0:
    #                print("X: ",xx[i][i1][:-1],xx[i][i2][:-1])
    #                print("Y: ",y1,y1,min(y1-y2),"\n")
    
    ## checking differences
    for i1,s1 in enumerate(asfnames[:-1]):
        for i2,s2 in enumerate(asfnames[i1+1:]):
            print("\n*******************")
            print(s1+" - "+s2)
            # difference between y values
            fig,ax=plt.subplots(figsize=(8,6))
            ax.set_title("Y")
            ax.hist([np.linalg.norm(yy[i1][j]-yy[i2][j])
                        for j in range(len(yy[i1]))
                    ],bins=20,range=(0,0.000004)
                    )
            plt.show()
            # difference between x values
            fig,ax=plt.subplots(figsize=(8,6))
            ax.set_title("X")
            ax.hist([np.linalg.norm(xx[i1][j][:-1]-xx[i2][j][:-1])
                        for j in range(len(xx[i1]))
                    ],bins=20,range=(0,0.000004)
                    )
            plt.show()
            
            
            