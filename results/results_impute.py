import numpy as np
import argparse
import scipy.sparse
import sys
sys.path.append('../')
from util_function import *
from benchmark_util import *
from R_util import generateLouvainCluster
from sklearn.cluster import KMeans 

#Evaluating imputing results
#Used to postprocess results of imputation
parser = argparse.ArgumentParser(description='Imputation Results')
parser.add_argument('--datasetName', type=str, default='4.Yan',
                    help='databaseName')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--regulized-type', type=str, default='Graph',
                    help='regulized type (default: Graph), otherwise: noregu')
parser.add_argument('--npyDir', type=str, default='../npyImpute/',
                    help='directory of npy')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--reconstr', type=str, default='',
                    help='iteration of imputed recon (default: '') alternative: 0,1,2')
parser.add_argument('--pcaNum', type=int, default=100,
                    help='Number of principle components (default: 100)')
# if have benchmark: use cell File
parser.add_argument('--benchmark',action='store_true', default=False, help="whether have benchmark")
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/allBench/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv',help="label Filename")
parser.add_argument('--n-clusters', default=20, type=int, help='number of clusters, 7 for cora, 6 for citeseer, 11 for 5.Pollen, 20 for MMP')

# if use part of cells Can be delete later
parser.add_argument('--cellFilename', type=str,default='/home/wangjue/biodata/scData/11.Kolodziejczyk.cellname.txt',help="cell Filename")
parser.add_argument('--cellIndexname',type=str,default='/home/wangjue/myprojects/scGNN/data/sc/11.Kolodziejczyk/ind.11.Kolodziejczyk.cellindex.txt',help="cell index Filename")

args = parser.parse_args()

if args.benchmark:
    labelFilename = args.labelFilename
    cellFilename  = args.cellFilename
    cellIndexFilename = args.cellIndexname
    true_labels = readTrueLabelList(labelFilename)

featuresOriginal = load_data(args.datasetName, args.discreteTag)
discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
## features         = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npy')
## features         = scipy.sparse.load_npz(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npz')
#Note: features is not actually used here
#  features         = load_sparse_matrix(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npz').tolil()
features         = None
dropi            = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+args.ratio+'_dropi.npy')
dropj            = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+args.ratio+'_dropj.npy')
dropix           = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+args.ratio+'_dropix.npy')

featuresImpute   = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+args.ratio+'_recon'+args.reconstr+'.npy')
l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

def imputeResult(inputData):
    '''
    Clustering on Imputed results function
    Here we both use Louvain(Use edge information) and Kmeans(No edge information) to do the clustering
    '''
    if type(inputData) is scipy.sparse.lil.lil_matrix:
        inputData = scipy.sparse.lil.lil_matrix.todense(inputData)
    if inputData.shape[0] > args.pcaNum:
        z,_ = pcaFunc(inputData, n_components=args.pcaNum)
    else:
        z = inputData
    _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
    listResult,size = generateLouvainCluster(edgeList)
    # modularity = calcuModularity(listResult, edgeList)
    # print('{:.4f}'.format(modularity))
    silhouette, chs, dbs = measureClusteringNoLabel(z, listResult)
    print('{:.4f} {:.4f} {:.4f} '.format(silhouette, chs, dbs), end='')
    if args.benchmark:
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
# imputeResult(features)
print('')

