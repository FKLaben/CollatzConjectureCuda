from functools import lru_cache



from numba import jit, cuda 
# to measure exec time 

from timeit import default_timer as timer


#@lru_cache(maxsize=None)
def collatz(value):
    if value%2==0:
        return value / 2
    else:
        return 3*value +1

# function optimized to run on gpu 
@jit(target_backend='cuda')						 
def collatz_recurs(value):
    if value <= 1:
        return 1
    else:
        #print(value)
        if value%2==0:
            return value / 2
        else:
            return 3*value +1


def collatz_recurs2(value):
    if value <= 1:
        return 1
    else:
        #print(value)
        return collatz_recurs2(collatz(value))


if __name__=="__main__":

    max = 500000
    
    start = timer()
    for value in range(max):
        returnValue = collatz_recurs(value)
    print("with GPU:", timer()-start) 
 
    start = timer() 
    for value in range(max):
        returnValue = collatz_recurs2(value)
    print("without GPU:", timer()-start)	



#for value in range(5000):
#    value = collatz_recurs(value)
#    print(value)

# value = collatz(value)
# print(value)
# value = collatz(value)
# print(value)
# value = collatz(value)
# print(value)
# value = collatz(value)