import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# sys.path.append('../')
import numpy as np
from util_function import *
from graph_function import *
import argparse

parser = argparse.ArgumentParser(description='main benchmark for scRNA with timer and mem')
#Benchmark related
parser.add_argument('--benchmark', type=str, default='/home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv',
                    help='the benchmark file of celltype (default: /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv)')
parser.add_argument('--input', type=str, default='filename',
                    help='input filename')
parser.add_argument('--inputOri', type=str, default='filename',
                    help='input filename')
args = parser.parse_args()

#Benchmark
bench_pd=pd.read_csv(args.benchmark,index_col=0)
bench_celltype=bench_pd.iloc[:,0].to_numpy()


#'saucie/13.txt'
z_pd = pd.read_csv(args.input,header=None)
listResult = z_pd.iloc[:,0].to_numpy()
zOut = np.load(args.inputOri,allow_pickle=True)
silhouette, chs, dbs = measureClusteringNoLabel(zOut, listResult)
ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(bench_celltype, listResult)
resultstr = str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs)
print(resultstr)
