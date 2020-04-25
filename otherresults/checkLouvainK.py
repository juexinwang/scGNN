import sys
sys.path.append('../')
import time
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import resource
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans,SpectralClustering,AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,FeatureAgglomeration,OPTICS,MeanShift
from model import AE, VAE, VAE2d
from util_function import *
from graph_function import *
from benchmark_util import *
from gae_embedding import GAEembedding,measure_clustering_results,test_clustering_benchmark_results
from R_util import generateLouvainCluster


parser = argparse.ArgumentParser(description='Test imputation from other imputation results')
parser.add_argument('--datasetName', type=str, default='481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                    help='For 10X: folder name of 10X dataset; For CSV: csv file name')
parser.add_argument('--datasetDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='Directory of dataset: default(/home/wangjue/biodata/scData/10x/6/)')                  
parser.add_argument('--LTMGDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--ltmgExpressionFile', type=str, default='Use_expression.csv',
                    help='expression File after ltmg in csv')
parser.add_argument('--ltmgFile', type=str, default='LTMG_sparse.mtx',
                    help='expression File in csv. (default:LTMG_sparse.mtx for sparse mode/ ltmg.csv for nonsparse mode) ')
parser.add_argument('--outputDir', type=str, default='npyGraphTest/',
                    help='save npy results in directory')
parser.add_argument('--nonsparseMode', action='store_true', default=False, 
                    help='SparseMode for running for huge dataset')

#Speed related
parser.add_argument('--batch-size', type=int, default=12800, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--Regu-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train in Regulatory Autoencoder (default: 500)')
parser.add_argument('--EM-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train in process of iteration EM (default: 200)')
parser.add_argument('--celltype-epochs', type=int, default=200, metavar='N',
                    help='number of epochs in celltype training (default: 200)')
parser.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                    help='number of iteration in total EM iteration (default: 10)')
parser.add_argument('--quickmode', action='store_true', default=False,
                    help='whether use quickmode, (default: no quickmode)')

#Regulation autoencoder
parser.add_argument('--regulized-type', type=str, default='LTMG',
                    help='regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--model', type=str, default='AE',
                    help='VAE/AE (default: AE)')
parser.add_argument('--gammaPara', type=float, default=0.1,
                    help='regulized parameter (default: 0.1)')
parser.add_argument('--regularizePara', type=float, default=0.9,
                    help='regulized parameter (default: 0.9)')
parser.add_argument('--L1Para', type=float, default=0.0,
                    help='L1 regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.0,
                    help='L2 regulized parameter (default: 0.001)')

#Build cell graph
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type (default: euclidean)')

#Graph Autoencoder
parser.add_argument('--useGAEembedding', action='store_true', default=False, 
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False, 
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--GAEmodel', type=str, default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')

#Clustering related
parser.add_argument('--n-clusters', default=20, type=int, help='number of clusters if predifined for KMeans/Birch ')
parser.add_argument('--clustering-method', type=str, default='LouvainK',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStats',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStats)')
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)') 
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')

# Converge related
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge-type', type=str, default='celltype',
                    help='type of converge condition: celltype/graph/both/either (default: either) ')
parser.add_argument('--converge-graphratio', type=float, default=0.01,
                    help='converge condition: ratio of graph ratio change in EM iteration (default: 0.01), 0-1')
parser.add_argument('--converge-celltyperatio', type=float, default=0.99,
                    help='converge condition: ratio of cell type change in EM iteration (default: 0.99), 0-1')

# dealing with zeros in imputation results
parser.add_argument('--zerofillFlag', action='store_true', default=False, 
                    help='fill zero or not before EM process (default: False)')
parser.add_argument('--noPostprocessingTag', action='store_false', default=True, 
                    help='whether postprocess imputated results, default: (True)') 
parser.add_argument('--postThreshold', type=float, default=0.01, 
                    help='Threshold to force expression as 0, default:(0.01)')  

# Debug
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saveinternal', action='store_true', default=True, 
                    help='whether save internal npy results or not')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')                    
parser.add_argument('--EMreguTag', action='store_true', default=False,
                    help='whether regu in EM process')
parser.add_argument('--debugMode', type=str, default='noDebug',
                    help='savePrune/loadPrune for debug reason (default: noDebug)')
parser.add_argument('--parallelLimit', type=int, default=0,
                    help='Number of cores usage for parallel pruning, 0 for use all cores (default: 0)')
#Benchmark related
parser.add_argument('--benchmark', type=str, default='/home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv',
                    help='the benchmark file of celltype (default: /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv)')
parser.add_argument('--npyFile', type=str, default='/storage/htc/joshilab/wangjue/no_dropout_recon/scimpute/13.Zeisel_recon.npy',
                    help='reconnpy')
parser.add_argument('--outFile', type=str, default='t.txt',
                    help='out file')
parser.add_argument('--resultFile', type=str, default='t.txt',
                    help='out file')

args = parser.parse_args()

#Benchmark
bench_pd=pd.read_csv(args.benchmark,index_col=0)
#t1=pd.read_csv('/home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv',index_col=0)
bench_celltype=bench_pd.iloc[:,0].to_numpy()

zOut = np.load(args.npyFile)
adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k), parallelLimit=args.parallelLimit)

if zOut.shape[0]< 2000:
    resolution = 0.8
else:
    resolution = 0.5

listResult,size = generateLouvainCluster(edgeList)
k = len(np.unique(listResult))
# print('Louvain cluster: '+str(k))
k = int(k*resolution) if k>3 else 2
clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
listResult = clustering.predict(zOut)

silhouette, chs, dbs = measureClusteringNoLabel(zOut, listResult)
ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(bench_celltype, listResult)
# print(str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs))
outstr = str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs)

with (open(args.outFile,"a+")) as fw:
    fw.write(outstr+'\n')   
    fw.close()  

with (open(args.resultFile,"w")) as fw:
    fw.writelines(listResult)  
    fw.close()    