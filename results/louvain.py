import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
from util_function import *
from graph_function import *
from R_util import generateLouvainCluster
import argparse

parser = argparse.ArgumentParser(description='main benchmark for scRNA with timer and mem')
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThreadNoPrune',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStats)')
#Benchmark related
parser.add_argument('--benchmark', type=str, default='/home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv',
                    help='the benchmark file of celltype (default: /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv)')
parser.add_argument('--input', type=str, default='filename',
                    help='input filename')
parser.add_argument('--output', type=str, default='filename',
                    help='input filename')
args = parser.parse_args()

#Benchmark
bench_pd=pd.read_csv(args.benchmark,index_col=0)
bench_celltype=bench_pd.iloc[:,0].to_numpy()

zOut = np.load(args.input,allow_pickle=True)
zOut,re = pcaFunc(zOut, n_components=10)
adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k))
listResult,size = generateLouvainCluster(edgeList)
silhouette, chs, dbs = measureClusteringNoLabel(zOut, listResult)
ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(bench_celltype, listResult)
resultstr = str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs)
print(resultstr)

with open(args.output,'w') as fw:
    fw.writelines("%s\n" % strr for strr in listResult)
