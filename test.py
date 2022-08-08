import numpy as np
from timeit import default_timer as timer
from numba import vectorize, cuda

@vectorize(["float32(float32,float32)"],target='cuda')
def VectorAdd(a,b):
    return a+b

def main():
    N=32000000
    A=np.ones(N,dtype=np.float32)
    B=np.ones(N,dtype=np.float32)
    C=np.zeros(N,dtype=np.float32)

    start =timer()
    VectorAdd(A,B,C)
    timee = timer()-start
    print(timee)
main()
print("hui")
