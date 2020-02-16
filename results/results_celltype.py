import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 1
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
from gae.model import GCNModelVAE, GCNModelAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
from deepWalk.skipGram import SkipGram
from sklearn.cluster import KMeans,SpectralClustering,AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,FeatureAgglomeration,MeanShift,OPTICS 
from clustering_metric import clustering_metrics
from tqdm import tqdm
from graph_function import *
from benchmark_util import *
from gae_embedding import *

# Evaluating celltype identification results
# Ref codes from https://github.com/MysteryVaibhav/RWR-GAE
parser = argparse.ArgumentParser()
parser.add_argument('--datasetName', type=str, default='MMPbasal',
                    help='databaseName')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--regulized-type', type=str, default='Graph',
                    help='regulized type (default: Graph), otherwise: noregu')
parser.add_argument('--npyDir',   type=str,default='../npyGraph10/',help="npyDir")
parser.add_argument('--reconstr', type=str, default='',
                    help='iteration of imputed recon (default: '') alternative: 0,1,2')
# if have benchmark: use cell File
parser.add_argument('--benchmark',action='store_true', default=False, help="whether have benchmark")
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/allBenchmark/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv',help="label Filename")
# if use part of the results, can be delete later
parser.add_argument('--cellFilename', type=str,default='/home/wangjue/biodata/scData/11.Kolodziejczyk.cellname.txt',help="cell Filename")
parser.add_argument('--cellIndexname',type=str,default='/home/wangjue/myprojects/scGNN/data/sc/11.Kolodziejczyk/ind.11.Kolodziejczyk.cellindex.txt',help="cell index Filename")
# paramters
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type (default: euclidean)')
parser.add_argument('--pcaNum', type=int, default=100,
                    help='Number of principle components (default: 100)')
# GAE related
parser.add_argument('--GAEmodel', type=str, default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--n-clusters', default=20, type=int, help='number of clusters, 7 for cora, 6 for citeseer, 11 for 5.Pollen, 20 for MMP')
args = parser.parse_args()

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'

if args.benchmark:
    labelFilename = args.labelFilename
    cellFilename  = args.cellFilename
    cellIndexFilename = args.cellIndexname
    true_labels = readTrueLabelList(labelFilename)

print("Original PCA")
originalFile = '../data/sc/{}/{}.features.csv'.format(args.datasetName,args.datasetName)
x = pd.read_csv(originalFile,header=None)
# PCA only number of cells larger than args.pcaNum, aka 100
if x.shape[0]>args.pcaNum:
    x, re = pcaFunc(x, n_components=args.pcaNum)
# para = 'euclidean:10'
adj, edgeList = generateAdj(x, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k))

xDiscret = x>np.mean(x,axis=0)
xDiscret = 1.0*xDiscret
xGAE=GAEembedding(xDiscret, adj, args)

if args.benchmark:
    test_clustering_benchmark_results(xGAE, edgeList, true_labels, args)
    test_clustering_benchmark_results(x, edgeList, true_labels, args)
else:
    test_clustering_results(xGAE, edgeList, args)
    test_clustering_results(x, edgeList, args)

print("Proposed Method")
z = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_z'+args.reconstr+'.npy')
# para = 'euclidean:10'
adj, edgeList = generateAdj(z, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k))

if args.benchmark:
    test_clustering_benchmark_results(z, edgeList, true_labels, args)
else:
    test_clustering_results(z, edgeList, args)

