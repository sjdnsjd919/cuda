import numpy as np
from timeit import default_timer as timer
from numba import vectorize
#6.78 s for 1e7
#vectprize uses scalar n input 1 output only
@vectorize(["float32(float32,float32)"],target='cpu')#1st is output tpe, then input, default is 1 instatce in GPU
def vectorAdd(a,b):
    return a+b

def main():
    N=10000000
    a=np.ones(N,dtype=np.float32)
    b=np.ones(N,dtype=np.float32)
    c=np.zeros(N,dtype=np.float32)
    starttime = timer()
    c= vectorAdd(a,b)
    timetot= timer()-starttime
    print(str(c[:5]))
    print(str(c[-5:]))
    print("time:" + str(timetot))
if __name__ == '__main__':
    main()
