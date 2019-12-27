import numpy as np
import argparse
import scipy.sparse
from util_function import *
from benchmark_util import * 

#Used to postprocess results of imputation
parser = argparse.ArgumentParser(description='Imputation Results')
parser.add_argument('--datasetName', type=str, default='MMPbasal',
                    help='databaseName')
parser.add_argument('--discreteTag', type=bool, default=False,
                    help='False/True')
parser.add_argument('--regulized-type', type=str, default='noregu',
                    help='regulized type (default: Graph), otherwise: noregu')
parser.add_argument('--npyDir', type=str, default='/home/wangjue/myprojects/scGNN/npyImpute/',
                    help='directory of npy')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
args = parser.parse_args()

featuresOriginal = load_data(args.datasetName, args.discreteTag)
discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
features         = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_features.npy')
dropi            = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropi.npy')
dropj            = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropj.npy')
dropix           = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_dropix.npy')

featuresImpute   = np.load(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+args.ratio+'_recon.npy')

l1Error = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)

print(l1Error)

