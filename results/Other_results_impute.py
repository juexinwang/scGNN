import numpy as np
import argparse
import scipy.sparse
import sys
sys.path.append('../')
from util_function import *
from benchmark_util import *
from R_util import generateLouvainCluster
from sklearn.cluster import KMeans 

#Evaluating imputing results from other methods
# Note: It is slightly different from results_impute.py
#Used to postprocess results of imputation
parser = argparse.ArgumentParser(description='Imputation Results')
parser.add_argument('--datasetName', type=str, default='MMPbasal',
                    help='databaseName')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--npyDir', type=str, default='../otherResults/SAUCIE_I/',
                    help='directory of npy')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--clusterTag', action='store_true', default=False,
                    help='whether the method has clusters (default: False)')
# if have benchmark: use cell File
parser.add_argument('--benchmark',action='store_true', default=False, help="whether have benchmark")
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/AnjunBenchmark/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv',help="label Filename")
parser.add_argument('--cellFilename', type=str,default='/home/wangjue/biodata/scData/11.Kolodziejczyk.cellname.txt',help="cell Filename")
parser.add_argument('--cellIndexname',type=str,default='/home/wangjue/myprojects/scGNN/data/sc/11.Kolodziejczyk/ind.11.Kolodziejczyk.cellindex.txt',help="cell index Filename")
parser.add_argument('--n-clusters', default=20, type=int, help='number of clusters, 7 for cora, 6 for citeseer, 11 for 5.Pollen, 20 for MMP')

args = parser.parse_args()

if args.benchmark:
    labelFilename = args.labelFilename
    cellFilename  = args.cellFilename
    cellIndexFilename = args.cellIndexname
    true_labels = readTrueLabelList(labelFilename, cellFilename, cellIndexFilename)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

features         = None
featuresImpute   = np.load(args.npyDir+datasetNameStr+'_'+args.ratio+'_recon.npy')
featuresOriginal = np.load(args.npyDir+datasetNameStr+'_'+args.ratio+'_featuresOriginal.npy')
dropi            = np.load(args.npyDir+datasetNameStr+'_'+args.ratio+'_dropi.npy')
dropj            = np.load(args.npyDir+datasetNameStr+'_'+args.ratio+'_dropj.npy')
dropix           = np.load(args.npyDir+datasetNameStr+'_'+args.ratio+'_dropix.npy')


def imputation_error_(X_mean, X, X_zero, i, j, ix):
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
    result = np.abs(x - y)
    return np.mean(result), np.median(result), np.min(result), np.max(result)

print(datasetNameStr)
l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error_(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

def imputeResult(inputData):
    '''
    Clustering on Imputed results function
    Here we both use Louvain(Use edge information) and Kmeans(No edge information) to do the clustering
    '''
    if type(inputData) is scipy.sparse.lil.lil_matrix:
        inputData = scipy.sparse.lil.lil_matrix.todense(inputData)
    z,_ = pcaFunc(inputData)
    _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
    listResult,size = generateLouvainCluster(edgeList)
    if args.benchmark:
        # Louvain
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(true_labels, listResult)
        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(ari, ami, nmi, cs, fms, vms, hs), end='')
        # KMeans
        clustering = KMeans(n_clusters=args.n_clusters, random_state=0).fit(z)
        listResult = clustering.predict(z)
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(true_labels, listResult)
        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(ari, ami, nmi, cs, fms, vms, hs), end='')
    else:
        # modularity = calcuModularity(listResult, edgeList)
        # print('{:.4f}'.format(modularity))
        silhouette, chs, dbs = measureClusteringNoLabel(z, listResult)
        print('{:.4f} {:.4f} {:.4f} '.format(silhouette, chs, dbs), end='')
    

imputeResult(featuresImpute)
imputeResult(featuresOriginal)

if args.benchmark:
    if args.clusterTag:
        clusters         = np.load(args.npyDir+datasetNameStr+'_'+args.ratio+'_clusters.npy')
        # Methods provided
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(true_labels, clusters)
        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(ari, ami, nmi, cs, fms, vms, hs), end='')

print('')

