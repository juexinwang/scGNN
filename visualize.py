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
from graph_function import * 

parser = argparse.ArgumentParser(description='Plot scRNA Results')
parser.add_argument('--datasetName', type=str, default='MPPbasal_allgene',
                    help='MPPbasal')
parser.add_argument('--dataset', type=str, default='MPPbasal_allgene_noregu_z.npy',
                    help='MPPbasal_noregu_z5.npy  data/sc/MPPbasal/MPPbasal.features.csv  /home/wangjue/scRNA/VarID_analysis/pca.csv')
parser.add_argument('--csvheader', type=bool, default=False,
                    help='Only for csv')
parser.add_argument('--saveFlag', type=bool, default=True,
                    help='save fig or not')
parser.add_argument('--saveDir', type=str, default='fig/',
                    help='save fig or not')
parser.add_argument('--npyDir', type=str, default='npy/',
                    help='save npy results in directory')
args = parser.parse_args()

if args.dataset[-3:] == 'npy':
    # z = np.load('MPPbasal_noregu_z5.npy')
    z = np.load(args.npyDir+args.dataset)
elif args.dataset[-3:] == 'csv':
    if args.csvheader==False:
        # z = pd.read_csv('data/sc/MPPbasal/MPPbasal.features.csv',header=None)
        z = pd.read_csv(args.dataset,header=None)
    else:
        # z = pd.read_csv('/home/wangjue/scRNA/VarID_analysis/pca.csv')
        z = pd.read_csv(args.dataset)
        z = z.to_numpy()
        z = z.transpose()

# Use all
# z = pd.read_csv('data/sc/MPPbasal/MPPbasal.features.csv',header=None)
# z = np.load('MPPbasal_noregu_z5.npy')
# z = pd.read_csv('/home/wangjue/scRNA/VarID_analysis/pca.csv')
# z = z.to_numpy()
# z = z.transpose()
# df['Cluster']= memberList

#PCA
def pcaFunc(z=z, n_components=100):
    pca = PCA(n_components=100)
    pca_result = pca.fit_transform(z)
    re = pd.DataFrame()
    re['pca-one'] = pca_result[:,0]
    re['pca-two'] = pca_result[:,1] 
    re['pca-three'] = pca_result[:,2]
    # re['Cluster'] = df['Cluster']
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return pca_result, re

#find Cluster from Louvain
def generateCluster(edgeList):
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
    
#UMAP
def drawUMAP(z,listResult,size):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(z)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=listResult, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(int(size))-0.5).set_ticks(np.arange(int(size)))
    plt.title('UMAP projection', fontsize=24)
    if args.saveFlag:
        plt.savefig(args.saveDir+args.dataset+'_UMAP.jpeg',dpi=300)

#Spring plot drawing
def drawSPRING(edgeList, listResult):
    G = nx.Graph()
    G.add_weighted_edges_from(edgeList)
    pos = nx.spring_layout(G)
    partition={}
    count = 0
    for item in listResult:
        partition[item] = item
        count += 1
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()
    if args.saveFlag:
        plt.savefig(args.saveDir+args.dataset+'_SPRING.jpeg',dpi=300)


# T-SNE
def drawTSNE(z, listResult):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(z)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['Cluster'] = listResult
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
    if args.saveFlag:
        plt.savefig(args.saveDir+args.dataset+'_TSNE.jpeg',dpi=300)

def drawFractPlot(exFile, geneFile, markerGeneList, listResult):
    expressionData = pd.read_csv(exFile,header=None)
    expressionData = expressionData.to_numpy()

    markerGeneIndexList = []
    geneDict = {}
    geneList = []

    # with open("data/sc/{}/{}.gene.txt".format(args.datasetName, args.datasetName), 'r') as f:
    with open(geneFile, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            line = line.strip()
            geneList.append(line)
            geneDict[line]=count
            count += 1
    f.close()

    for markerGene in markerGeneList:
        if markerGene in geneDict:
            markerGeneIndexList.append(geneDict[markerGene])
        else:
            print('Cannot find '+markerGene+' in gene.txt')

    # dim: [4394, 20]
    useData = expressionData[:,markerGeneIndexList]
    zData = stats.zscore(useData,axis=1)
    allIndexm1  = np.where(np.less(zData,-1.0))
    allIndexm01 = np.where(np.logical_and(np.greater_equal(zData,-1.0),np.less(zData,0.0)))
    allIndex01 = np.where(np.logical_and(np.greater_equal(zData,0.0),np.less(zData,1.0)))
    allIndex13 = np.where(np.logical_and(np.greater_equal(zData,1.0),np.less(zData,3.0)))
    allIndex3 = np.where(np.greater_equal(zData,3.0))
    allIndex1 = np.where(np.greater_equal(zData,1.0))

    allIndex = allIndex3

    # resultTablem1 = [[0.0] * len(markerGeneList)  for i in range(len(set(listResult)))]
    resultTable = [[0.0] * len(markerGeneList)  for i in range(len(set(listResult)))]
    resultTableRatio = [[0.0] * len(markerGeneList)  for i in range(len(set(listResult)))]

    clusterNum = [0 for i in range(len(set(listResult)))]
    for i in range(useData.shape[0]):
        clusterNum[listResult[i]] += 1

    clusterNum = np.asarray(clusterNum).reshape(len(set(listResult)),1)

    for i in np.arange(allIndex[0].shape[0]):
        clusterIndex = listResult[allIndex[0][i]]
        resultTable[clusterIndex][allIndex[1][i]] += 1

    resultTableUsage = resultTable/clusterNum

    df = pd.DataFrame(data=resultTableUsage,index=range(len(set(listResult))),columns=markerGeneList)
    ax = sns.heatmap(df)
    plt.savefig(args.saveDir+args.dataset+'_MarkerGenes.jpeg',dpi=300)

# Main plots:
#pca_result, re = pcaFunc(z, n_components=100)

# edgeList = np.load('MPPbasal_noregu_edgeList1.npy')

#_, edgeList = generateAdj(pca_result, graphType='Thresholdgraph', para = 'cosine:0.95')
_, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
# _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'cosine:10')
# _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'correlation:10')

listResult,size = generateCluster(edgeList)

# drawUMAP(z,listResult,size)

# drawSPRING(edgeList, listResult)
# drawTSNE(z, listResult)

# test marker genes:
markerGeneList = ['Kit','Flt3','Dntt','Ebf1','Cd19','Lmo4','Ms4a2','Ear10','Cd74','Irf8','Mpo','Elane','Ngp','Mpl','Pf4','Car1','Gata1','Hbb-bs','Ptgfrn','Mki67']
exFile = 'data/sc/{}/{}.features.csv'.format(args.datasetName, args.datasetName)
geneFile = 'data/sc/{}/{}.gene.txt'.format(args.datasetName, args.datasetName)
drawFractPlot(exFile, geneFile, markerGeneList, listResult)


# new={}
# for i in range(len(part)):
#     for j in range(len(part[i])):
#         new[part[i][j]]=i



