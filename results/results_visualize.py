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
from sklearn.metrics import silhouette_samples, silhouette_score
from graph_function import *
from benchmark_util import * 

# Visualize results of celltype identifications
parser = argparse.ArgumentParser(description='Plot scRNA Results')
parser.add_argument('--datasetName', type=str, default='MMPbasal_all',
                    help='MPPbasal')
parser.add_argument('--dataset', type=str, default='MPPbasal_noregu_z2.npy',
                    help='MPPbasal_noregu_z.npy  ../data/sc/MPPbasal/MPPbasal.features.csv  /home/wangjue/scRNA/VarID_analysis/pca.csv')
parser.add_argument('--csvheader', type=bool, default=False,
                    help='Only for csv')
parser.add_argument('--saveFlag', type=bool, default=True,
                    help='save fig or not')
parser.add_argument('--saveDir', type=str, default='../fig/',
                    help='save fig or not')
parser.add_argument('--npyDir', type=str, default='../npyGraph10/',
                    help='save npy results in directory')
parser.add_argument('--benchFlag', type=bool, default=True,
                    help='True for data with benchmark')
args = parser.parse_args()

# Use all
# z = pd.read_csv('data/sc/MPPbasal/MPPbasal.features.csv',header=None)
# z = np.load('MPPbasal_noregu_z5.npy')
# z = pd.read_csv('/home/wangjue/scRNA/VarID_analysis/pca.csv')
# z = z.to_numpy()
# z = z.transpose()
# df['Cluster']= memberList


# Main plots:
# edgeList = np.load('MPPbasal_noregu_edgeList1.npy')

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
    
    #PCA
    z, re = pcaFunc(z, n_components=100)

_, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
# _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'cosine:10')
# _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'correlation:10')

listResult,size = generateCluster(edgeList)

# drawUMAP(z, listResult, size, args.saveDir, args.dataset, args.saveFlag)

# drawSPRING(edgeList, listResult, args.saveDir, args.dataset, args.saveFlag)
# drawTSNE(z, listResult, args.saveDir, args.dataset, args.saveFlag)

# # test marker genes:
# markerGeneList = ['Kit','Flt3','Dntt','Ebf1','Cd19','Lmo4','Ms4a2','Ear10','Cd74','Irf8','Mpo','Elane','Ngp','Mpl','Pf4','Car1','Gata1','Hbb-bs','Ptgfrn','Mki67']
# exFile = 'data/sc/{}/{}.features.csv'.format(args.datasetName, args.datasetName)
# geneFilename = args.datasetName
# #_LTMG use same genefiles as original
# if args.datasetName[-4:]=='LTMG':
#     geneFilename = args.datasetName[:-5]
# geneFile = 'data/sc/{}/{}.gene.txt'.format(geneFilename, geneFilename)
# drawFractPlot(exFile, geneFile, markerGeneList, listResult, args.saveDir, args.dataset, args.saveFlag)

# modularity = calcuModularity(listResult, edgeList)
# print('{:.4f}'.format(modularity))
silhouette, chs, dbs = measureClusteringNoLabel(z, listResult)
print('{:.4f} {:.4f} {:.4f}'.format(silhouette, chs, dbs))

if args.benchFlag:
    labelFilename = '/home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_cell_label.csv'
    cellFilename  = '/home/wangjue/biodata/scData/5.Pollen.cellname.txt'
    cellIndexFilename = '/home/wangjue/myprojects/scGNN/data/sc/5.Pollen/ind.5.Pollen.cellindex.txt'
    truelabel = readTrueLabelList(labelFilename, cellFilename, cellIndexFilename)

    ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(truelabel, listResult)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(ari, ami, nmi, cs, fms, vms, hs))


# new={}
# for i in range(len(part)):
#     for j in range(len(part[i])):
#         new[part[i][j]]=i



