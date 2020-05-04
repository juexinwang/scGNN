# Analysis using MAGIC method
import magic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
sys.path.append('../')
from benchmark_util import impute_dropout
from util_function import *
from benchmark_util import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='12.Klein',help='12.Klein')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--regulized-type', type=str, default='LTMG',
                    help='regulized type (default: LTMG), otherwise: noregu')
parser.add_argument('--replica', type=str, default='1',
                    help='replica')
parser.add_argument('--n-clusters', type=int, default=4,
                    help='replica')
parser.add_argument('--npyDir', type=str, default='../npyImputeG2E_LK',
                    help='directory of npy')
parser.add_argument('--labelFilename',type=str,default='/home/jwang/data/scData/12.Klein/Klein_cell_label.csv',help="label Filename")

args = parser.parse_args()

labelFilename = args.labelFilename
true_labels = readTrueLabelList(labelFilename)

featuresOriginal = load_data(args.datasetName, False)
ori = featuresOriginal.todense()
oriz = ori.copy()

# for 0.3/0.6/0.9
# npyDir = args.npyDir+'_'+args.ratio+'_'+args.replica+'/'
npyDir = args.npyDir+'/'

dropi            = np.load(npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropi.npy')
dropj            = np.load(npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropj.npy')
dropix           = np.load(npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropix.npy')

for item in dropix:
	oriz[dropi[item],dropj[item]]=0.0

# Add log transformation
x = np.log(oriz+1)

# Load single-cell RNA-seq data
# Default is KNN=5
magic_operator = magic.MAGIC()
# magic_operator = magic.MAGIC(knn=10)
X_magic = magic_operator.fit_transform(x, genes="all_genes")
recon = X_magic

# np.savetxt(npyDir+str(args.datasetName)+'_'+str(args.ratio)+'_'+str(args.replica)+'_recon.csv',recon,delimiter=",",fmt='%10.4f')
np.savetxt('MAGIC_'+str(args.datasetName)+'_'+str(args.ratio)+'_'+str(args.replica)+'_recon.csv',recon,delimiter=",",fmt='%10.4f')
np.savetxt('MAGIC_'+str(args.datasetName)+'_'+str(args.ratio)+'_'+str(args.replica)+'_ori.csv',x,delimiter=",",fmt='%10.4f')


features         = None
featuresImpute   = recon
l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

def imputeResult(inputData):
    '''
    Clustering on Imputed results function
    Here we both use Louvain(Use edge information) and Kmeans(No edge information) to do the clustering
    '''
    if type(inputData) is scipy.sparse.lil.lil_matrix:
        inputData = scipy.sparse.lil.lil_matrix.todense(inputData)
    # if inputData.shape[0] > args.pcaNum:
    z,_ = pcaFunc(inputData, n_components=15)
    # else:
    #     z = inputData
    _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
    listResult,size = generateLouvainCluster(edgeList)
    # modularity = calcuModularity(listResult, edgeList)
    # print('{:.4f}'.format(modularity))
    silhouette, chs, dbs = measureClusteringNoLabel(z, listResult)
    print('{:.4f} {:.4f} {:.4f} '.format(silhouette, chs, dbs), end='')
    # if args.benchmark:
    # Louvain
    ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(true_labels, listResult)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(ari, ami, nmi, cs, fms, vms, hs), end='')
    # KMeans
    clustering = KMeans(n_clusters=args.n_clusters, random_state=0).fit(z)
    listResult = clustering.predict(z)
    ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(true_labels, listResult)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(ari, ami, nmi, cs, fms, vms, hs), end='')

imputeResult(featuresImpute)
imputeResult(featuresOriginal)
print('')
