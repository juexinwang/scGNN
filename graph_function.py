from scipy.spatial import distance_matrix, minkowski_distance, distance
import scipy.sparse
import sys
import pickle
import csv
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
import time
from multiprocessing import Pool
import multiprocessing 

# Calculate graph, return adjcency matrix in 0/1
def generateAdj(featureMatrix, graphType='KNNgraph', para = None, parallelLimit = 0, adjTag = True ):
    """
    Generating edgeList 
    """
    edgeList = None
    adj = None

    if graphType == 'KNNgraphPairwise':
        edgeList = calculateKNNgraphDistanceMatrixPairwise(featureMatrix, para)
    elif graphType == 'KNNgraph':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrix(featureMatrix, distanceType=distanceType, k=k)
    elif graphType == 'Thresholdgraph':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            threshold = float(parawords[1])
        edgeList = calculateThresholdgraphDistanceMatrix(featureMatrix, distanceType=distanceType, threshold=threshold)
    elif graphType == 'KNNgraphThreshold':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
            threshold = float(parawords[2])
        edgeList = calculateKNNThresholdgraphDistanceMatrix(featureMatrix, distanceType=distanceType, k=k, threshold=threshold)
    elif graphType == 'KNNgraphML':
        # with weights!
        # https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
        # https://scikit-learn.org/stable/modules/outlier_detection.html
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrixML(featureMatrix, distanceType=distanceType, k=k)
    elif graphType == 'KNNgraphStats':
        # with weights!
        # with stats, one std is contained
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrixStats(featureMatrix, distanceType=distanceType, k=k, parallelLimit=parallelLimit)
    elif graphType == 'KNNgraphStatsSingleThread':
        # with weights!
        # with stats, one std is contained, but only use single thread
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType=distanceType, k=k)
    else:
        print('Should give graphtype')

    if adjTag:
        graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    
    return adj, edgeList

# Calculate graph, return adjcency matrix in weighted
def generateAdjWeighted(featureMatrix, graphType='KNNgraph', para = None, parallelLimit = 0, outAdjTag = True ):
    """
    outAdjTag: saving space for not generating adj for giant network without GAE 
    """
    edgeListWeighted = None
    adj = None

    if graphType == 'KNNgraphStats':
        # with weights!
        # with stats, one std is contained
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeListWeighted = calculateKNNgraphDistanceMatrixStatsWeighted(featureMatrix, distanceType=distanceType, k=k, parallelLimit=parallelLimit)
    elif graphType == 'KNNgraphStatsSingleThread':
        # with weights!
        # with stats, one std is contained, but only use single thread
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeListWeighted = calculateKNNgraphDistanceMatrixStatsSingleThreadWeighted(featureMatrix, distanceType=distanceType, k=k)
    else:
        print('Should give graphtype')
    
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeListWeighted)
    adj = nx.adjacency_matrix(Gtmp)
    
    return adj, edgeListWeighted

#para: measuareName:k
def calculateKNNgraphDistanceMatrixPairwise(featureMatrix, para):
    r"""
    KNNgraphPairwise:  measuareName:k
    Pairwise:5
    Minkowski-Pairwise:5:1
    """
    measureName = ''
    k = 5
    if para != None:
        parawords = para.split(':')
        measureName = parawords[0]        

    distMat = None
    if measureName == 'Pairwise':
        distMat = distance_matrix(featureMatrix,featureMatrix)
        k = int(parawords[1])
    elif measureName == 'Minkowski-Pairwise':
        p = int(parawords[2])
        distMat = minkowski_distance(featureMatrix,featureMatrix,p=p)
        k = int(parawords[1])        
    else:
        print('meausreName in KNNgraph does not recongnized')
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j]))
    
    return edgeList

#para: measuareName:k
def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    r"""
    KNNgraph: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    distanceType incude:
    Distance functions between two numeric vectors u and v. Computing distances over a large collection of vectors is inefficient for these functions. Use pdist for this purpose.

    braycurtis(u, v[, w])	Compute the Bray-Curtis distance between two 1-D arrays.
    canberra(u, v[, w])	Compute the Canberra distance between two 1-D arrays.
    chebyshev(u, v[, w])	Compute the Chebyshev distance.
    cityblock(u, v[, w])	Compute the City Block (Manhattan) distance.
    correlation(u, v[, w, centered])	Compute the correlation distance between two 1-D arrays.
    cosine(u, v[, w])	Compute the Cosine distance between 1-D arrays.
    euclidean(u, v[, w])	Computes the Euclidean distance between two 1-D arrays.
    jensenshannon(p, q[, base])	Compute the Jensen-Shannon distance (metric) between two 1-D probability arrays.
    mahalanobis(u, v, VI)	Compute the Mahalanobis distance between two 1-D arrays.
    minkowski(u, v[, p, w])	Compute the Minkowski distance between two 1-D arrays.
    seuclidean(u, v, V)	Return the standardized Euclidean distance between two 1-D arrays.
    sqeuclidean(u, v[, w])	Compute the squared Euclidean distance between two 1-D arrays.
    wminkowski(u, v, p, w)	Compute the weighted Minkowski distance between two 1-D arrays.

    Distance functions between two boolean vectors (representing sets) u and v. As in the case of numerical vectors, pdist is more efficient for computing the distances between all pairs.

    dice(u, v[, w])	Compute the Dice dissimilarity between two boolean 1-D arrays.
    hamming(u, v[, w])	Compute the Hamming distance between two 1-D arrays.
    jaccard(u, v[, w])	Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.
    kulsinski(u, v[, w])	Compute the Kulsinski dissimilarity between two boolean 1-D arrays.
    rogerstanimoto(u, v[, w])	Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.
    russellrao(u, v[, w])	Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.
    sokalmichener(u, v[, w])	Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.
    sokalsneath(u, v[, w])	Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.
    yule(u, v[, w])	Compute the Yule dissimilarity between two boolean 1-D arrays.

    hamming also operates over discrete numerical vectors.
     
    """       

    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j]))
    
    return edgeList

#para: measuareName:threshold
def calculateThresholdgraphDistanceMatrix(featureMatrix, distanceType='euclidean', threshold=0.5):
    r"""
    Thresholdgraph: Graph with certain threshold 
    """       

    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        indexArray = np.where(distMat[i,:]>threshold)
        for j in indexArray[0]:
            edgeList.append((i,j))
    
    return edgeList

#para: measuareName:k:threshold
def calculateKNNThresholdgraphDistanceMatrix(featureMatrix, distanceType='cosine', k=10, threshold=0.5):
    r"""
    Thresholdgraph: KNN Graph with certain threshold 
    """       

    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k-1):
            if (distMat[i,res[j]]>threshold):
                edgeList.append((i,res[j]))
        # edgeList.append((i,res[k-1]))
    
    return edgeList


#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixML(featureMatrix, distanceType='euclidean', k=10, param=None):
    r"""
    Thresholdgraph: KNN Graph with Machine Learning based methods

    IsolationForest
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest 
    """       

    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
    edgeList=[]

    # parallel: n_jobs=-1 for using all processors
    clf = IsolationForest( behaviour = 'new', contamination= 'auto', n_jobs=-1)

    for i in np.arange(distMat.shape[0]):
        res = distMat[i,:].argsort()[:k+1]
        preds = clf.fit_predict(featureMatrix[res,:])       
        for j in np.arange(1,k+1):
            # weight = 1.0
            if preds[j]==-1:
                weight = 0.0
            else:
                weight = 1.0
            #preds[j]==-1 means outliner, 1 is what we want
            edgeList.append((i,res[j],weight))
    
    return edgeList

#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10, param=None):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """       

    edgeList=[]
    # Version 1: cost memory, precalculate all dist

    ## distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
    ## parallel
    # distMat = pairwise_distances(featureMatrix,featureMatrix, distanceType, n_jobs=-1)
    
    # for i in np.arange(distMat.shape[0]):
    #     res = distMat[:,i].argsort()[:k+1]
    #     tmpdist = distMat[res[1:k+1],i]
    #     mean = np.mean(tmpdist)
    #     std = np.std(tmpdist)
    #     for j in np.arange(1,k+1):
    #         if (distMat[i,res[j]]<=mean+std) and (distMat[i,res[j]]>=mean-std):
    #             weight = 1.0
    #         else:
    #             weight = 0.0
    #         edgeList.append((i,res[j],weight))

    ## Version 2: for each of the cell, calculate dist, save memory 
    p_time = time.time()
    for i in np.arange(featureMatrix.shape[0]):
        if i%10000==0:
            print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')
        tmp=featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,featureMatrix, distanceType)
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist)
        for j in np.arange(1,k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0,res[0][j]]<=boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((i,res[0][j],weight))

    # Version 3: for each of the cell, calculate dist, use heapq to accelerate
    # However, it cannot defeat sort
    # Get same results as this article
    # https://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently
    #
    # p_time = time.time()
    # for i in np.arange(featureMatrix.shape[0]):
    #     if i%10000==0:
    #         print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')
    #     tmp=featureMatrix[i,:].reshape(1,-1)
    #     distMat = distance.cdist(tmp,featureMatrix, distanceType)[0]
    #     # res = distMat.argsort()[:k+1]
    #     res = heapq.nsmallest(k+1, range(len(distMat)), distMat.take)[1:k+1]
    #     tmpdist = distMat[res]
    #     boundary = np.mean(tmpdist)+np.std(tmpdist)
    #     for j in np.arange(k):
    #         # TODO: check, only exclude large outliners
    #         # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
    #         if distMat[res[j]]<=boundary:
    #             weight = 1.0
    #         else:
    #             weight = 0.0
    #         edgeList.append((i,res[j],weight))
    
    return edgeList

# kernelDistance
def kernelDistance(distance,delta=1.0):
    '''
    Calculate kernel distance
    '''
    kdist = np.exp(-distance/2*delta**2)
    return kdist

#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStatsSingleThreadWeighted(featureMatrix, distanceType='euclidean', k=10, param=None):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods weighted, SingleThread version
    """       

    edgeListWeighted=[]

    ## Version 2: for each of the cell, calculate dist, save memory 
    p_time = time.time()
    for i in np.arange(featureMatrix.shape[0]):
        if i%10000==0:
            print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')
        tmp=featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,featureMatrix, distanceType)
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist)
        for j in np.arange(1,k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0,res[0][j]]<=boundary:
                weight = kernelDistance(distMat[0,res[0][j]])
                edgeListWeighted.append((i,res[0][j],weight))
            # else: not add weights
    
    return edgeListWeighted

class FindKParallel():
    '''
    A class to find K parallel
    '''
    def __init__(self,featureMatrix,distanceType,k):
        self.featureMatrix = featureMatrix
        self.distanceType = distanceType
        self.k = k

    def vecfindK(self,i):
        '''
        Find topK in paral
        '''
        edgeList_t=[]
        # print('*'+str(i))
        tmp=self.featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,self.featureMatrix, self.distanceType)
        # print('#'+str(distMat))
        res = distMat.argsort()[:self.k+1]
        # print('!'+str(res))
        tmpdist = distMat[0,res[0][1:self.k+1]]
        # print('@'+str(tmpdist))
        boundary = np.mean(tmpdist)+np.std(tmpdist)
        # print('&'+str(boundary))
        for j in np.arange(1,self.k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0,res[0][j]]<=boundary:
                weight = kernelDistance(distMat[0,res[0][j]])
                edgeList_t.append((i,res[0][j],weight))
        # print('%'+str(len(edgeList_t)))
        return edgeList_t
    
    def work(self):
        return Pool().map(self.vecfindK, range(self.featureMatrix.shape[0]))


#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStats(featureMatrix, distanceType='euclidean', k=10, param=None, parallelLimit=0):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods using parallel cores
    """       
    edgeList=[]
    # Get number of availble cores
    USE_CORES = 0 
    NUM_CORES = multiprocessing.cpu_count()
    # if no limit, use all cores
    if parallelLimit == 0:
        USE_CORES = NUM_CORES
    # if limit < cores, use limit number
    elif parallelLimit < NUM_CORES:
        USE_CORES = parallelLimit
    # if limit is not valid, use all cores
    else:
        USE_CORES = NUM_CORES
    print('Start Pruning using '+str(USE_CORES)+' of '+str(NUM_CORES)+' available cores') 

    t= time.time()
    #Use number of cpus for top-K finding
    with Pool(USE_CORES) as p:
        # edgeListT = p.map(vecfindK, range(featureMatrix.shape[0]))
        edgeListT = FindKParallel(featureMatrix, distanceType, k).work()

    t1=time.time()
    print('Pruning succeed in '+str(t1-t)+' seconds')
    flatten = lambda l: [item for sublist in l for item in sublist]   
    t2=time.time()
    edgeList = flatten(edgeListT)    
    print('Prune out ready in '+str(t2-t1)+' seconds')
       
    return edgeList

#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStatsWeighted(featureMatrix, distanceType='euclidean', k=10, param=None, parallelLimit=0):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods using parallel cores
    """       
    edgeListWeighted=[]
    # Get number of availble cores
    USE_CORES = 0 
    NUM_CORES = multiprocessing.cpu_count()
    # if no limit, use all cores
    if parallelLimit == 0:
        USE_CORES = NUM_CORES
    # if limit < cores, use limit number
    elif parallelLimit < NUM_CORES:
        USE_CORES = parallelLimit
    # if limit is not valid, use all cores
    else:
        USE_CORES = NUM_CORES
    print('Start Pruning using '+str(USE_CORES)+' of '+str(NUM_CORES)+' available cores') 

    t= time.time()
    #Use number of cpus for top-K finding
    with Pool(USE_CORES) as p:
        # edgeListT = p.map(vecfindK, range(featureMatrix.shape[0]))
        edgeListT = FindKParallel(featureMatrix, distanceType, k).work()

    t1=time.time()
    print('Pruning succeed in '+str(t1-t)+' seconds')
    flatten = lambda l: [item for sublist in l for item in sublist]   
    t2=time.time()
    edgeListWeighted = flatten(edgeListT)    
    print('Prune out ready in '+str(t2-t1)+' seconds')
       
    return edgeListWeighted


# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict

# Function originates from old file
# For cell,use feature matrix as input, row as cells, col as genes
# Load gold standard edges into sparse matrix
# No edge types
# output mtx, tfDict
# Additional outfile for matlab
def read_edge_file_csc(edgeList, nodesize, k=5):
    row=[]
    col=[]
    data=[]
    
    for edge in edgeList:
        row.append(edge[0])
        col.append(edge[1])
        data.append(1.0)
        row.append(edge[1])
        col.append(edge[0])
        data.append(1.0)

    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(nodesize, nodesize))
    
    #python output
    # return mtx, tfDict

    #Output for matlab
    return mtx, row, col, data

# Function originates from old file
# genereate graph dict
def read_edge_file_dict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict

