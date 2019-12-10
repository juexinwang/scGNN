from __future__ import print_function
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
import pandas as pd
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import community
from graph_function import * 

# Use all
# z = pd.read_csv('data/sc/MPPbasal/MPPbasal.features.csv',header=None)
z = np.load('MPPbasal_noregu_z5.npy')
# z = pd.read_csv('/home/wangjue/scRNA/VarID_analysis/pca.csv')
# z = z.to_numpy()
# z = z.transpose()
# df['Cluster']= memberList



#PCA
pca = PCA(n_components=100)
pca_result = pca.fit_transform(z)
re = pd.DataFrame()
re['pca-one'] = pca_result[:,0]
re['pca-two'] = pca_result[:,1] 
re['pca-three'] = pca_result[:,2]
# re['Cluster'] = df['Cluster']
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# edgeList = np.load('MPPbasal_noregu_edgeList1.npy')

#_, edgeList = generateAdj(pca_result, graphType='Thresholdgraph', para = 'cosine:0.95')
_, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
# _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'cosine:10')
# _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'correlation:10')

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
    count+= 1
    


#UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(z)
plt.scatter(embedding[:, 0], embedding[:, 1], c=listResult, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(int(size))-0.5).set_ticks(np.arange(int(size)))
plt.title('UMAP projection', fontsize=24)


# new={}
# for i in range(len(part)):
#     for j in range(len(part[i])):
#         new[part[i][j]]=i


#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))

nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()


# T-SNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['Cluster'] = df['Cluster']
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", 
    y="tsne-2d-two",
    hue="Cluster",
    palette=sns.color_palette("brg",int(size)),
    data=df_subset,
    legend="full",
    # alpha=0.3
)


