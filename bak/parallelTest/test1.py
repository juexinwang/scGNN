from multiprocessing import Pool
import numpy as np
import time
from scipy.spatial import distance_matrix, minkowski_distance, distance
import resource

k =10
distanceType = 'euclidean'
def vecfindK(i):
    edgeList_t=[]
    # print('*'+str(i))
    tmp=featureMatrix[i,:].reshape(1,-1)
    distMat = distance.cdist(tmp,featureMatrix, distanceType)
    # print('#'+str(distMat))
    res = distMat.argsort()[:k+1]
    # print('!'+str(res))
    tmpdist = distMat[0,res[0][1:k+1]]
    # print('@'+str(tmpdist))
    boundary = np.mean(tmpdist)+np.std(tmpdist)
    # print('&'+str(boundary))
    for j in np.arange(1,k+1):
        # TODO: check, only exclude large outliners
        # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
        if distMat[0,res[0][j]]<=boundary:
            weight = 1.0
        else:
            weight = 0.0
        edgeList_t.append((i,res[0][j],weight))
    # print('%'+str(len(edgeList_t)))
    return edgeList_t

featureMatrix = np.random.normal(size=(30000, 32))
F = np.zeros((featureMatrix.shape[0], ))
# print(featureMatrix)
mem1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# print('Mem1 consumption: '+str(mem))
t= time.time()
# with Pool() as p:
# with Pool(2) as p:
with Pool(processes=4,maxtasksperchild=1) as p:
    edgeListT = p.map(vecfindK, range(featureMatrix.shape[0]))

mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print('Mem consumption: '+str(mem)+' '+str(mem-mem1))
t1=time.time()
flatten = lambda l: [item for sublist in l for item in sublist]
t2=time.time()
edgeList = flatten(edgeListT)
print(str(t1-t))
print(str(t2-t1))
print(Pool().)

