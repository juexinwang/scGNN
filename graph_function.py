from scipy.spatial import distance_matrix
import scipy.sparse
import sys
import pickle
import csv
import networkx as nx

#Graph related functions
#graph Edge
class graphEdge:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# Calculate graph, return adjcency matrix
def generateAdj(featureMatrix, graphType='KNNgraph', para = None):
    edgeList = None
    if graphType == None:
        print('Should give graphtype')
    elif graphType == 'KNNgraph':
        edgeList = calculateKNNgraphDistanceMatrix(featureMatrix, para)
    
    graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    
    return adj

#para: measuareName:k
def calculateKNNgraphDistanceMatrix(featureMatrix, para):
    r"""
    KNNgraph:  measuareName:k
    Eucledian-Pairwise:5
    """
    measureName = ''
    k = 5
    if para != None:
        parawords = para.split(':')
        measureName = parawords[0]
        k = int(parawords[1])

    distMat = None
    if measureName == 'Eucledian-Pairwise':
        distMat = distance_matrix(featureMatrix.todense(),featureMatrix.todense())
    else:
        print('meausreName in KNNgraph does not recongnized')
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

