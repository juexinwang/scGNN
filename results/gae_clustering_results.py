from __future__ import division
from __future__ import print_function
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
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

# Ref codes from https://github.com/MysteryVaibhav/RWR-GAE
parser = argparse.ArgumentParser()
parser.add_argument('--npyDir',type=str,default='../npyGraph10/',help="npyDir")
parser.add_argument('--zFilename',type=str,default='5.Pollen_all_noregu_recon0.npy',help="z Filename")
parser.add_argument('--benchmark',type=bool,default=True,help="whether have benchmark")
# cell File
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_cell_label.csv',help="label Filename")
parser.add_argument('--cellFilename',type=str,default='/home/wangjue/biodata/scData/5.Pollen.cellname.txt',help="cell Filename")
parser.add_argument('--cellIndexname',type=str,default='/home/wangjue/myprojects/scGNN/data/sc/5.Pollen_all/ind.5.Pollen_all.cellindex.txt',help="cell index Filename")
parser.add_argument('--originalFile',type=str,default='../data/sc/5.Pollen_all/5.Pollen_all.features.csv',help="original csv Filename")
args = parser.parse_args()

if args.benchmark:
    labelFilename = args.labelFilename
    cellFilename  = args.cellFilename
    cellIndexFilename = args.cellIndexname
    true_labels = readTrueLabelList(labelFilename, cellFilename, cellIndexFilename)

print("Proposed")
z = np.load(args.npyDir+args.zFilename)
adj, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')

print("Start GAE")
zDiscret = z>np.mean(z,axis=0)
zDiscret = 1.0*zDiscret
zGAE=GAEembedding(zDiscret, adj)

print("GAE clustering")
if args.benchmark:
    test_clustering_benchmark_results(zGAE, edgeList, true_labels)
else:
    test_clustering_results(zGAE, edgeList)

print("Original PCA")
x = pd.read_csv(args.originalFile,header=None)
x, re = pcaFunc(x, n_components=100)
adj, edgeList = generateAdj(x, graphType='KNNgraphML', para = 'euclidean:10')
if args.benchmark:
    test_clustering_benchmark_results(x, edgeList, true_labels)
else:
    test_clustering_results(x, edgeList)

print("Before GAE")
if args.benchmark:
    test_clustering_benchmark_results(z, edgeList, true_labels)
else:
    test_clustering_results(x, edgeList)