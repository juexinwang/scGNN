from multiprocessing import Pool
import numpy as np
import time

def foo(x): 
    print(x)
    # print('*')
    # print(x.shape)
    # return [i+1.0 for i in x]
    return [i+1.0 for i in featureMatrix[x,:]]

featureMatrix = np.random.normal(size=(10, 3))
F = np.zeros((featureMatrix.shape[0], ))
print(featureMatrix)
t= time.time()

with Pool() as p:
    # F=p.map(foo, featureMatrix )
    F=p.map(foo, range(10) )

# t1=time.time()
# print(str(t1-t))
# F1=np.asarray(F)
# t2=time.time()
# print(str(t2-t1))
print(F)
