# Script to test efficiency of louvain

# Option 1: Original version, use r version of louvain, it takes time to link R, and need install rpy2.
# Not use anymore
# Clustering is different between Case one and two 
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

# case one:
edgeList = []
edgeList.append((0,2,1.0))
edgeList.append((1,2,1.0))
edgeList.append((2,3,1.0))
edgeList.append((3,4,1.0))
edgeList.append((4,5,1.0))
edgeList.append((4,6,1.0))

# case two:
edgeList.append((0,2,1.0))
edgeList.append((1,2,1.0))
edgeList.append((2,3,0.1))
edgeList.append((3,4,1.0))
edgeList.append((4,5,1.0))
edgeList.append((4,6,1.0))

fromVec = []
toVec   = []
weightVec = []
for edge in edgeList:
    fromVec.append(edge[0])
    toVec.append(edge[1])
    weightVec.append(edge[2])

igraph = importr('igraph')
base   = importr('base')
fromV  = ro.FloatVector(fromVec)
toV    = ro.FloatVector(toVec)
# weightV= ro.FloatVector([0.1,1.0,1.0,0.1,1.0])
weightV= ro.FloatVector(weightVec)
links  = ro.DataFrame({'from':fromV,'to':toV,'weight':weightV})
g  = igraph.graph_from_data_frame(links,directed = False)
cl = igraph.cluster_louvain(g)

def as_dict(vector):
    """Convert an RPy2 ListVector to a Python dict"""
    result = {}
    for i, name in enumerate(vector.names):
        if isinstance(vector[i], ro.ListVector):
            result[name] = as_dict(vector[i])
        elif len(vector[i]) == 1:
            result[name] = vector[i][0]
        else:
            result[name] = vector[i]
    return result

cl_dict = as_dict(cl)
df = pd.DataFrame()
# df['Cluster']=cl_dict['membership']
size = float(len(set(cl_dict['membership'])))

listResult=[]
count = 0
for i in range(len(cl_dict['membership'])):
    listResult.append(int(cl_dict['membership'][i])-1)
    count += 1

# Option 2: use package python-louvain, but does not work
# Clustering is identical between Case one and two, so we cannot use it
import networkx as nx
import community as community_louvain
G = nx.Graph()
G.add_weighted_edges_from(edgeList)
partition = community_louvain.best_partition(G,weight='weight')


# Option 3: use igraph, pure python and looks right
# Clustering is identical between Case one and two, so we cannot use it
import numpy as np
from igraph import *
#Case 1:
W=np.zeros((7,7))
W[0,2]=1.0
W[1,2]=1.0
W[2,3]=1.0
W[3,4]=1.0
W[4,5]=1.0
W[4,6]=1.0

#Case 2:
W=np.zeros((7,7))
W[0,2]=1.0
W[1,2]=1.0
W[2,3]=0.1
W[3,4]=1.0
W[4,5]=1.0
W[4,6]=1.0

graph = Graph.Weighted_Adjacency(W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
louvain_partition = graph.community_multilevel(weights=graph.es['weight'], return_levels=False)
print(louvain_partition)

