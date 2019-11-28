from scipy.spatial import distance_matrix
import scipy.sparse
import sys
import pickle
import csv

#Original version of generating cell graph
class KNNEdge:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# Calculate KNN graph, return row and col
def cal_distanceMatrix(featureMatrix, k=5):
    distMat = distance_matrix(featureMatrix.todense(),featureMatrix.todense())
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append(KNNEdge(i,res[j]))
    
    return edgeList


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