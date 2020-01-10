from __future__ import print_function
import argparse
from scipy.spatial import distance_matrix, minkowski_distance, distance
import scipy.sparse
import sys
import pickle
import csv
import networkx as nx
import numpy as np
import time
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature
import scipy
from scipy import stats
import pandas as pd
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import community

# Reorganize the utils for R dependency, get louvain out here for some platforms cannot use R

#find Cluster from Louvain
def generateLouvainCluster(edgeList):
    # no weights
    # G = nx.Graph(edgeList)

    # weighted edges: networkx,does not work
    # https://github.com/vtraag/louvain-igraph
    # https://python-louvain.readthedocs.io/en/latest/api.html
    # G = nx.Graph()
    # G.add_weighted_edges_from(edgeList)
    # partition = community.best_partition(G,weight='weight')
    # valueResults = []
    # for key in partition.keys():
    #     valueResults.append(partition[key])

    # df = pd.DataFrame()
    # df['Cluster']=valueResults

    # R:
    # https://github.com/dgrun/RaceID3_StemID2_package/blob/master/R/VarID_functions.R
    fromVec = []
    toVec   = []
    weightVec = []
    for edge in edgeList:
        fromVec.append(edge[0])
        toVec.append(edge[1])
        weightVec.append(edge[2])

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate()

    igraph = importr('igraph')
    base   = importr('base')
    fromV  = ro.FloatVector(fromVec)
    toV    = ro.FloatVector(toVec)
    # weightV= ro.FloatVector([0.1,1.0,1.0,0.1])
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

    return listResult, size