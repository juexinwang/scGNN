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
from sklearn.metrics import *
from sklearn.metrics.cluster import *
from graph_function import *

#PCA
def pcaFunc(z, n_components=100):
    pca = PCA(n_components=100)
    pca_result = pca.fit_transform(z)
    re = pd.DataFrame()
    re['pca-one'] = pca_result[:,0]
    re['pca-two'] = pca_result[:,1] 
    re['pca-three'] = pca_result[:,2]
    # Not print Now
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
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
def drawUMAP(z,listResult,size,saveDir,dataset,saveFlag=True):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(z)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=listResult, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(int(size))-0.5).set_ticks(np.arange(int(size)))
    plt.title('UMAP projection', fontsize=24)
    if saveFlag:
        plt.savefig(saveDir+dataset.split('/')[-1]+'_UMAP.jpeg',dpi=300)

#Spring plot drawing
def drawSPRING(edgeList, listResult, saveDir, dataset, saveFlag=True):
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
                                    node_color = str(count / len(set(partition.values()))))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()
    if saveFlag:
        plt.savefig(saveDir+dataset.split('/')[-1]+'_SPRING.jpeg',dpi=300)


# T-SNE
def drawTSNE(z, listResult, saveDir, dataset, saveFlag=True):
    size = len(set(listResult))
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
    if saveFlag:
        plt.savefig(saveDir+dataset.split('/')[-1]+'_TSNE.jpeg',dpi=300)

def drawFractPlot(exFile, geneFile, markerGeneList, listResult, saveDir, dataset, saveFlag=True):
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
    zData = stats.zscore(useData,axis=0)

    resultTable = [[0.0] * len(markerGeneList)  for i in range(len(set(listResult)))]
    resultTableRatio = [[0.0] * len(markerGeneList)  for i in range(len(set(listResult)))]

    clusterNum = [0 for i in range(len(set(listResult)))]

    for i in range(zData.shape[0]):
        clusterIndex = listResult[i]
        clusterNum[clusterIndex] += 1
        for j in range(zData.shape[1]):           
            resultTable[clusterIndex][j] += zData[i,j]
    
    clusterNum = np.asarray(clusterNum).reshape(len(set(listResult)),1)

    resultTableUsage = resultTable/clusterNum

    clusterSortDict={}
    clusterSortList=[]
    for i in range(resultTableUsage.shape[1]):
        indexArray = np.argsort(resultTableUsage[:,i],axis=0)[::-1]
        for j in indexArray:
            if not j in clusterSortDict:
                clusterSortList.append(j)
                clusterSortDict[j]=0
                break

    df = pd.DataFrame(data=resultTableUsage[clusterSortList,:], index=clusterSortList, columns=markerGeneList)
    ax = sns.heatmap(df,cmap="YlGnBu")
    if saveFlag:
        plt.savefig(saveDir+dataset.split('/')[-1]+'_MarkerGenes.jpeg',dpi=300)
    # np.save('resultTable.npy',resultTable)
    # np.save('resultTableUsage.npy',resultTableUsage)


def calcuModularity(listResult,edgeList):
    '''
    Calculate Modularity through networkx modularity
    https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python
    '''
    G = nx.Graph()
    G.add_weighted_edges_from(edgeList)
    partition={}
    for item in range(len(listResult)):
        partition[item] = listResult[item]
    global_modularity = community.modularity(partition, G)
    return global_modularity

def measureClusteringNoLabel(z, listResult):
    '''
    Measure clustering without labels
    return:
    silhouette, calinski_harabasz_score(Variance Ratio Criterion), davies_bouldin_score

    silhouette: most important
    davies_bouldin_score: lower the better, others: higher the better

    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
    '''
    silhouette = silhouette_score(z, listResult)
    chs = calinski_harabasz_score(z, listResult)
    dbs = davies_bouldin_score(z, listResult)
    return silhouette, chs, dbs


def measureClusteringTrueLabel(labels_true, labels_pred):
    '''
    Measure clustering with true labels
    return: 
    Adjusted Rand Index, Ajusted Mutual Information, Normalized Mutual Information, completeness score, fowlkes mallows score, v measure score, homogeneity score 

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html   
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html

    '''
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    cs  = completeness_score(labels_true, labels_pred)
    fms = fowlkes_mallows_score(labels_true, labels_pred)
    vms = v_measure_score(labels_true, labels_pred)
    hs  = homogeneity_score(labels_true, labels_pred)
    return ari, ami, nmi, cs, fms, vms, hs

# labelFilename:     /home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_cell_label.csv
# cellFilename:      /home/wangjue/biodata/scData/5.Pollen.cellname.txt
# cellIndexFilename: /home/wangjue/myprojects/scGNN/data/sc/5.Pollen/ind.5.Pollen.cellindex.txt
def readTrueLabelList(labelFilename, cellFilename, cellIndexFilename):
    '''
    Read gold standard label from file
    '''
    cellDict = {}
    count = -1
    with open(labelFilename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if count >= 0:
                line = line.strip()
                words = line.split(',')
                cellDict[words[0]]=int(words[1])-1
            count += 1
        f.close()
    
    cellIndexDict = {}
    count = -1
    with open(cellFilename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if count >= 0:
                line = line.strip()
                cellIndexDict[count]=line            
            count += 1
        f.close()
    
    labelList = []
    count = 0
    with open(cellIndexFilename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if int(line) in cellIndexDict:
                cellName = cellIndexDict[int(line)]
                memberName = cellDict[cellName]
            else:
                memberName = 100
            labelList.append(memberName)            
            count += 1
        f.close()

    return labelList


def impute_dropout(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = scipy.sparse.lil_matrix.copy(X)
    # select non-zero subset
    i,j = X_zero.nonzero()
    
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
       
    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix

# TODO
def impute_dropout_sparse(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # select non-zero subset
    i,j = np.nonzero(X_zero)
    
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
       
    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix



# IMPUTATION METRICS
# Ref:
# https://github.com/romain-lopez/scVI-reproducibility/blob/master/demo_code/benchmarking.py
def imputation_error(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need 
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    all_index = i[ix], j[ix]
    x = X_mean[all_index[0],all_index[1]]
    y =      X[all_index[0],all_index[1]]
    yuse = scipy.sparse.lil_matrix.todense(y)
    yuse = np.asarray(yuse).reshape(-1)
    result = np.abs(x - yuse)
    # return np.median(np.abs(x - yuse))
    return np.mean(result), np.median(result), np.min(result), np.max(result)

#TODO
def imputation_error_sparse(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need 
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x - y))
