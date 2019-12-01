from scipy.spatial import distance_matrix, minkowski_distance, distance
import scipy.sparse
import sys
import pickle
import csv
import networkx as nx
import numpy as np

#Graph related functions
#graph Edge
class graphEdge:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# Calculate graph, return adjcency matrix
def generateAdj(featureMatrix, graphType='KNNgraph', para = None):
    edgeList = None
    if graphType == 'KNNgraphPairwise':
        edgeList = calculateKNNgraphDistanceMatrixPairwise(featureMatrix, para)
    elif graphType == 'KNNgraph':
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
        edgeList = calculateKNNgraphDistanceMatrix(featureMatrix, distanceType=distanceType, k=k)
    else:
        print('Should give graphtype')
    
    graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    
    return adj

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
        distMat = distance_matrix(featureMatrix.todense(),featureMatrix.todense())
        k = int(parawords[1])
    elif measureName == 'Minkowski-Pairwise':
        p = int(parawords[2])
        distMat = minkowski_distance(featureMatrix.todense(),featureMatrix.todense(),p=p)
        k = int(parawords[1])        
    else:
        print('meausreName in KNNgraph does not recongnized')
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append(graphEdge(i,res[j]))
    
    return edgeList

#para: measuareName:k
def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=5):
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

    distMat = distance.cdist(featureMatrix.todense(),featureMatrix.todense(), distanceType)
        
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append(graphEdge(i,res[j]))
    
    return edgeList



# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge.row
        end2 = edge.col
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
        row.append(edge.row)
        col.append(edge.col)
        data.append(1.0)
        row.append(edge.col)
        col.append(edge.row)
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
        end1 = edge.row
        end2 = edge.col
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

