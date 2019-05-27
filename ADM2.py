### Declarations

import numpy as np
import copy

## DESDEO
from desdeo.method.NIMBUS import NIMBUS
from desdeo.optimization import SciPyDE
from desdeo.problem.toy import RiverPollution

from desdeo.preference import NIMBUSClassification

## Rectangles
from rtree import index as rindex

### Supplementary functions

## Calculate hypervolume of a box given by min and max points
def hv_box(mn,mx):
    return np.prod([mxi-mni for mni,mxi in zip(mn,mx)])

## Given a nested list, Returns a list of all lists of size k x 2
# for extracting results of the recursive box partitioning function divbox_rec
def flat_boxlist(a,k):
    # leaf of recursion calls
    try: # check if a is k x 2 list with non-list elements
        if len(a)==k and \
           any(
               (
                len(ai)==2 and \
                not( any( isinstance(aii,list) for aii in ai ) )
               ) for ai in a
              ):
            return [a]
    except:
        pass
    # next level recursion call
    if isinstance(a,list):
        return sum([flat_boxlist(i,k) for i in a],[])
    else:
        return []
            
## Given min and max vectors of a box as lists,
#  Returns the vector representation for rtree: 
#  [min_1,min_2,...,min_k,max_1,...,max_k]
def box2rindex(mn,mx):
    return mn+mx

## Given the rtree representation of a box as a list / numpy array
#       [min_1,min_2,...,min_k,max_1,...,max_k],
## Returns the list [[min vector], [max vector]] of the box 
def rindex2box(v):
    return np.array(v).reshape(2,-1).tolist()

## Recursive function for generating all open boxes partitioning a given box,
#  resulted from subtract the dual domination cone (represented by its vertex)
# Given: 
#  vrange = [for each i: [min,max] if the range of the part is defined for this i,
#           or [min,mid,max] if (mid=the vertex component) belongs to the open range
#           and the selection of higher or lower range is not defined for this i]
#  nlo = nr. of dimensions, for which the range is defined to be below the vertex component
#  nhi = nr. of dimensions, for which the range is defined to be above the vertex component
#  k = total nr. of dimensions
#  ii - currently considered dimension nr. (for previous ones, the range is defined)
# Returns:
#  if range is defined for all i (nlo+nhi==k) => [for each i, [min,max]] (leaf of recursion)
#  if some range is not defined (nlo+nhi<k) => branching the recursion, i.e. 
#       for j=(first index with undefined range), call divbox_rec in cases of upper and lower ranges
def divbox_rec(vrange,nlo,nhi,k,ii):
    #x#print(vrange,": ",nlo,"+",nhi,", ii=",ii)
    # If in each dimension, range of the considered part is defined => recursion leaf
    if nlo+nhi==k: 
        if nlo<k and nhi<k: # the part is not dominating / dominated by the vertex
            return [vrange]
        else:
            return []
    # For some dimension, the part can be divided into higher/lower parts
    # init. two branches of the considered part, to be passed to next recursion level
    vlo=copy.deepcopy(vrange) # undefined range -> lower range
    vhi=copy.deepcopy(vrange) # undefined range -> upper range
    for i in range(ii,k):
        if len(vrange[i])==3: # i = first index with undefined range
            vlo[i]=vrange[i][:2] # select lower range
            vhi[i]=vrange[i][1:] # select upper range
            # all parts of the box is concatenation of the two branches
            return divbox_rec(vlo,nlo+1,nhi,k,i+1) + \
                   divbox_rec(vhi,nlo,nhi+1,k,i+1)
                           

### Potential region structure for minimization problems based on rtree package class
#   Box ID (int) represent the ordinary number of the act of box creation
#   Additional attributes:
#    .ndim = space dimension
#    .nbox = number of boxes in the structure
#    .ncre = number of acts of boxes creation
#    ._hypervol = sum of hypervolume of boxes
class potreg(rindex.Index):
    
    def __init__(self,ideal,nadir):
        # setting the space dimension and passing to rtree in Property object
        ndim=len(ideal)
        p = rindex.Property()
        p.dimension = ndim
        # initializing object
        rindex.Index.__init__(self,properties=p)
        self.ndim = ndim
        # adding the first rectangle
        self.nbox = 1
        self.ncre = 1
        # the initial box = the Pareto range
        self.insert(1,box2rindex(ideal,nadir))
        self._hypervol=hv_box(ideal,nadir)
    
    # Given a vector v, Returns the list of boxes intersecting with 
    # the dual domination cone at v, presented in rlist format:
    # [id,[rlist mins-maxes vector]]
    def _pintersect(self,v):
        # box ids and coordinates are merged for checking uniqueness
        inter=[# intersection with the positive cone
            [b.id] + b.bbox for b in
            self.intersection(
                    box2rindex(v,[np.inf for i in range(self.ndim)]), 
                    objects=True
                    )
           ]+ \
           [# intersection with the negative cone
            [b.id] + b.bbox for b in
            self.intersection(
                    box2rindex([-np.inf for i in range(self.ndim)],v), 
                    objects=True
                    )
           ]
        # list of unique intersected boxes
        if len(inter)==0:
            h=[]
        else:
            h=np.unique(inter,axis=0).tolist()
        return [[int(i[0]),i[1:]] for i in h]  
    
    ## Given a vector v, transforms the potential region structure by taking
    # set differences between all boxes and the dual domination cone.
    # Returns True if the structure has changed
    def addpoint(self,v):
        # h is the list of boxes [id,[rlist min-max vector]] intersecting with cones
        h=self._pintersect(v)
        if len(h)==0:
            print("### No intersections! Boxes: ", self.nbox," of ",self.ncre)
            return False
        # init the indicator if the potential region changed
        qchange=False
        ## Consider all intersected boxes one-by-one
        for b in h:
            rid=b[0] # box ID
            rv=b[1] # box vector in rtree format
            ## creating the vector of ranges / ranges with a midpoint for the recursive function
            vrange=(np.array(rindex2box(rv)).T).tolist() # init. list of the box ranges 
            # init. nrs. of dimensions with defined lower and higher ranges, respectively
            nlo=0
            nhi=0
            vrange_rec=[] # init. the list for recursive function
            for i in range(self.ndim):
                if v[i]<=vrange[i][0]: # the range belongs to the higher part
                    vrange_rec.append(vrange[i])
                    nhi+=1
                elif v[i]>=vrange[i][1]: # the range belongs to the lower part
                    vrange_rec.append(vrange[i])
                    nlo+=1
                else: # the vertex point is inside the range => it is undefined
                    vrange_rec.append([vrange[i][0],v[i],vrange[i][1]])
            ## Consider different cases of box-cones intersection
            # Box is a subset of a cone => removed from the potential region
            if nlo==self.ndim or nhi==self.ndim:
                qchange=True
                self.nbox-=1
                self.delete(rid,rv)
                self._hypervol-=hv_box(*rindex2box(rv))
            # Box does not intersect with either of the cones => do nothing 
            elif nlo>0 and nhi>0:
                break
            # rest of cases: box is intersected => divide into parts
            else:
                qchange=True
                # remove the original box
                self.nbox-=1
                self.delete(rid,rv)
                self._hypervol-=hv_box(*rindex2box(rv))
                # insert its remaining parts
                newboxes=flat_boxlist(divbox_rec(vrange_rec,nlo,nhi,self.ndim,0),self.ndim)
                for c in newboxes:
                    self.nbox+=1
                    self.ncre+=1
                    self.insert(self.ncre,box2rindex(*(np.array(c).T.tolist())))
                    self._hypervol+=hv_box(*(np.array(c).T.tolist()))
        return qchange
    
    # Returns list of al boxes (as [ [[min vect.],[max vect.]],id ]) in the potential region 
    def boxes(self):
        return [[rindex2box(b.bbox),b.id]
                for b in self.intersection(
                    box2rindex(
                            [-np.inf for i in range(self.ndim)],
                            [np.inf for i in range(self.ndim)]
                                ), objects=True
                    )
                ]

### Automatic Decision Maker basic class representing ADM instance
# interacting with a method when solving a minimization problem.
# Input: one or more Pareto optimal objective vectors, 
# Output: two reference points (aspiration,reservation)
## Attributes
#   .k: nr. of objectives
#   .itern: current iteration nr.
#   .c: coefficient of optimizm (float)               
#   ._ideal, ._nadir: corresponding points
#   ._potreg: potential region based on potreg class
#   ._paretoset: list of nuique Pareto objective vectors
#   ._npareto: nr. of unique Pareto objective vectors
#   ._uf: utility function (R^k,Ideal,Nadir -> R)
#   .telemetry: dictionary of lists collecting relevant information in each iteration
## Methods
#   ._box_score: function (box=[min vector,max vector]) -> score (float)
#               which is used when selecting boxes
#   ._ufbox: basic example of _box_score calculating UF at the representative point
#   .box_pref: given a box, returns preference information related to this box
#   ._box_refpoint: basic example of box_pref returning [[min. point],[max. point]]
#           of the box which serve as aspiration and reservation ref. points               
#   ._upd: Given one or list of Pareto optimal objective vectors, 
#          adds new ones to the Pareto optimal set, updates the potential region
#          and returns [True iff potential region was changed, list of new Pareto optima]
#   .potboxes: returns the list of all boxes of the potential region
#   .bestbox: returns the best box [ [[min. point],[max.point]],id ] based on _box_score
#   .nextiter: Given one or set of Pareto optima, updates the potential region
#              and returns new preference information               



class ADM:
    def __init__(self,ideal,nadir,uf,coptimizm):
        self.k=len(ideal)
        self._ideal=ideal
        self._nadir=nadir
        self.itern=1
        self._potreg=potreg(ideal,nadir)
        self._paretoiter=[]
        self._paretoset=[]
        self._npareto=0
        self.c=coptimizm
        self._uf=uf
        self._box_score=self._ufbox
        self.telemetry={\
                "hypervol": [], # hypervolume of potential region after update
                "maxuf": [], # max. utility of newly obtained solutions
                "Pareto": [], # list of derived Pareto optimal objective in the iter.
                "nboxes": [], # nr. of boxes after update
                "crboxes": [], # nr. of boxes created so far (after the update)
                "npareto": [], # number of Pareto obj. vectors in the pool after update
                "bestbox": [], # the best box selected after update
                "pref": [] # preference information generated after update
                }
        
## Return hypervolume of boxes
    def hypervol(self):
        return self._potreg._hypervol

## Calculating UF at the representative point of a box (b=[min.v,max.v])
#  used in the basic version as the score function by default       
    def _ufbox(self,b):
        # calculate UF at the point alpha*min + (1-alpha)*max
        return self._uf(
                (np.array(b)*[[self.c],[1-self.c]]).sum(axis=0),
                self._ideal,self._nadir)

## Calculating scalar score of a box, used when selecting the best box,
# the higher score the better
    def _box_score(self,b):
        return self._ufbox(b)

## Returns [aspiration vect., reservation vect] for a given box
    def _box_refpoint(self,b):
        return [list(b[0]),list(b[1])]
## Returns preference information (wrapper)
    def box_pref(self,b):
        return self._box_refpoint(b)

## Given one or list of objective vectors, updates Pareto set and potential region;
#  Returns [whether potential region changed, the list of new Pareto objective vectors]
    def _upd(self,pp):
        if len(pp)==0:
            return [False,[]]
        if not(hasattr(pp[0],"__iter__")):
            pp=[pp]
        ## updating Pareto optimal set
        # updating telemetry
        self.telemetry["Pareto"].append(pp)
        ufmax=-np.inf
        for pi in pp:
            ufi=self._uf(pi,self._ideal,self._nadir)
            #x#print("norm: ",normalize(pi,self._ideal,self._nadir))
            #x#print("uf: ",self._uf(pi,self._ideal,self._nadir))
            if ufi>ufmax:
                ufmax=ufi
        self.telemetry["maxuf"].append(ufmax)
        # list of new Pareto solutions which are not in the pool
        pnew=[]
        for p in pp:
            qincl=True
            for p1 in self._paretoset:
                if p==p1:
                    qincl=False
                    break
            if qincl:
                pnew.append(p)
        if len(pnew)==0:
            return [False,[]]
        self._paretoset.extend(pnew)
        self._npareto+=len(pnew)
        # updating the potential region and calculating change indicator
        # list of results (if potreg changed) of adding all Pareto points
        qpoints=[self._potreg.addpoint(point) for point in pnew]
        return [any(qpoints),pnew]

## Returns the potential region as a list of boxes [min vect. , max vect.]
# ADM fatigue, memory etc. are modelled here 
    def potboxes(self):
        return self._potreg.boxes()

## Finds the best box based on _box_score and returns as [[min vect. , max vect.],id=ncre]
    def bestbox(self):
        return max(self.potboxes(),key=lambda b:self._box_score(b[0]))    

            
## Given one or list of objective vectors, updates the potential region and
#  Returns {
#           "pref": [asp. vect, reserv. vect], 
#           "boxid": creation nr. of the best box,
#           "nboxes": current nr. of boxes in potreg,
#           "npareto": current nr. of Pareto objective vectors,
#           "changed?" True if potreg changed
#           }
    def nextiter(self,p):
        ## updating the potential region and Pareto set
        upnew=self._upd(p)
        self.telemetry["hypervol"].append(self.hypervol())
        self.telemetry["nboxes"].append(self._potreg.nbox)
        self.telemetry["crboxes"].append(self._potreg.ncre)
        self.telemetry["npareto"].append(self._npareto)
        bb=self.bestbox()
        #print([normalize(x,self._ideal,self._nadir) for x in bb[0]])
        self.telemetry["bestbox"].append(bb)
        newpref=self.box_pref(bb[0])
        self.telemetry["pref"].append(newpref)
        self.itern+=1
        return {"pref": newpref,
                "changed?": upnew[0]
                }

### ADM class for Nimbus method

#? future features:
#    * koef (0,+inf), default=1 for putting temp. ref.point on the half-line
#      between best Pareto and representative point of best box
#    * adjust temp. ref. point components: 
#       o  greater than or close to ideal => "<" class
#       o  less than or close to nadir => ">" class
#       o  close to the best Pareto => "=" class
#
class ADM_Nimbus(ADM):
## Returns Nimbus-specific preference information
    def box_pref(self,b):
        return [("<=",x) for x in self._box_refpoint(b)[0]]


                    ##########
                    ## MAIN ##
                    ##########
                    
## General forms of parametric utility functions 
#  defined for maximization criteria in the region [0,1]^k
# CES based on multiplication
def CES_mult(xx,ww):
    return np.prod([x**w for x,w in zip(xx,ww)])
# CES based on power summation
def CES_sum(xx,ww,p):
    try:
        return sum([w*x**p for x,w in zip(xx,ww)])**(1/p)
    except:
        print("x: ",xx,", w: ",ww)
def UF_TOPSIS(xx,ww):
    d_NIS=sum([(w*(1-x))**2 for x,w in zip(xx,ww)])**(1/2)
    d_PIS=sum([(w*x)**2 for x,w in zip(xx,ww)])**(1/2)
    return d_NIS(d_NIS+d_PIS)

## Linear normalization: ideal -> 1, nadir -> 0
#  which converts minimization objectives to maximization objectives
def normalize(xx,ideal,nadir):
    return [(nad-x)/(nad-idl) for x,idl,nad in zip(xx,ideal,nadir)]




#############


## Instances of utility functions used in experiments with water treatment problem,
#  defined on [0,1]^k for maximization objectives
water_w1=[600,1,20,5000]
water_wmult=[5,1,1,0.01]
water_UFs=[
           lambda xx: CES_mult(xx,water_wmult),
           lambda xx: CES_sum(xx,water_w1,-3),
           lambda xx: UF_TOPSIS(xx,water_w1)
        ]
UFn=1 # choosing a UF from the list
coptimizm=1 # coefficient of optimizm  


            
problem = RiverPollution()
method = NIMBUS(problem, SciPyDE)
print("Ideal, nadir",problem.ideal,problem.nadir)


# in simpler version, pref.info does not depend on current solution(s)
# results = method.init_iteration()
A=ADM_Nimbus(
        problem.ideal,
        problem.nadir,
        lambda x,ideal,nadir: water_UFs[UFn](normalize(x,ideal,nadir)),
        1)
p=[]
for i in range(5):
    print("Iteration ",i)
    print("Created: ",A._potreg.ncre, ", left: ",A._potreg.nbox,", count: ",
          A._potreg.count(box2rindex([-np.inf for i in range(4)],
                                     [np.inf for i in range(4)]))
          )
    result=A.nextiter(p)
    pref=result["pref"]
    pref1=normalize([x[1] for x in pref],A._ideal,A._nadir)
    print("Preferences:",[format(x,"1.10") for x in pref1])
    p=[method._factories[0].result(
                      NIMBUSClassification(method, pref), None
                      )[1]
                                                     for i in range(1)]
    #print("Pareto:",[format(x,"1.10") for x in normalize(p[0],A._ideal,A._nadir)])
    #x# print("New: ",p)




