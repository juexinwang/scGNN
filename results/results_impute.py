import numpy as np
import argparse
import scipy.sparse
import sys
sys.path.append('../')
from util_function import *
from benchmark_util import *
from R_util import generateLouvainCluster 

#Evaluating imputing results
#Used to postprocess results of imputation
parser = argparse.ArgumentParser(description='Imputation Results')
parser.add_argument('--datasetName', type=str, default='MMPbasal',
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
# if have benchmark: use cell File
parser.add_argument('--benchmark',action='store_true', default=False, help="whether have benchmark")
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/AnjunBenchmark/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv',help="label Filename")
parser.add_argument('--cellFilename', type=str,default='/home/wangjue/biodata/scData/11.Kolodziejczyk.cellname.txt',help="cell Filename")
parser.add_argument('--cellIndexname',type=str,default='/home/wangjue/myprojects/scGNN/data/sc/11.Kolodziejczyk/ind.11.Kolodziejczyk.cellindex.txt',help="cell index Filename")

args = parser.parse_args()

if args.benchmark:
    labelFilename = args.labelFilename
    cellFilename  = args.cellFilename
    cellIndexFilename = args.cellIndexname
    true_labels = readTrueLabelList(labelFilename, cellFilename, cellIndexFilename)

featuresOriginal = load_data(args.datasetName, args.discreteTag)
discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
#TODO
# features         = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npy')
features         = load_sparse_matrix(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npz').tolil()
# features         = scipy.sparse.load_npz(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npz')
dropi            = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropi.npy')
dropj            = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropj.npy')
dropix           = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropix.npy')

featuresImpute   = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_recon'+args.reconstr+'.npy')
l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

def imputeResult(inputData):
    '''
    Impute results function
    '''
    # z = pd.read_csv('data/sc/MPPbasal/MPPbasal.features.csv',header=None)
    # z = pd.read_csv('data/sc/{}/{}.features.csv'.format(args.datasetName, args.datasetName),header=None)
    if type(inputData) is scipy.sparse.lil.lil_matrix:
        inputData = scipy.sparse.lil.lil_matrix.todense(inputData)
    z,_ = pcaFunc(inputData)
    _, edgeList = generateAdj(z, graphType='KNNgraphML', para = 'euclidean:10')
    listResult,size = generateLouvainCluster(edgeList)
    if args.benchmark:
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(true_labels, listResult)
        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(ari, ami, nmi, cs, fms, vms, hs), end='')
    else:
        # modularity = calcuModularity(listResult, edgeList)
        # print('{:.4f}'.format(modularity))
        silhouette, chs, dbs = measureClusteringNoLabel(z, listResult)
        print('{:.4f} {:.4f} {:.4f} '.format(silhouette, chs, dbs), end='')
    

imputeResult(featuresImpute)
imputeResult(featuresOriginal)
# imputeResult(features)
print('')

