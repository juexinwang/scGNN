import os
import numpy as np
import pandas as pd
import argparse
import scipy.sparse
import sys
sys.path.append('../')
from util_function import *
from benchmark_util import *
from R_util import generateLouvainCluster
from sklearn.cluster import KMeans
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods')
args = parser.parse_args()

# Notes:
# Call by submit_Impute_others.sh


datasetList = [
    '9.Chung',
    '11.Kolodziejczyk',
    '12.Klein',
    '13.Zeisel',
]

oridirStr = '../npyImputeG2E'
medirStr = '../'

seedList = ['1','2','3']
ratioList = [0.1,0.3,0.6,0.8]
methodList = ['magic','saucie','saver','scimpute','scvi','dca','deepimpute']

def outResults(datasetName,seed,ratio,method):
    featuresOriginal = load_data(datasetName, discreteTag=False)

    features         = None
    dropi            = np.load(oridirStr+'_'+seed+'/'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropi.npy')
    dropj            = np.load(oridirStr+'_'+seed+'/'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropj.npy')
    dropix           = np.load(oridirStr+'_'+seed+'/'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropix.npy')

    # scGNN results
    # featuresImpute   = np.load(npyDir+datasetName+'_'+args.regulized_type+discreteStr+'_'+args.ratio+'_10-0.1-0.9-0.0-0.3-'+args.regupara+'_recon'+args.reconstr+'.npy')
    featuresImpute   = np.load(medirStr+method+'/'+datasetName+'_'+ratio+'_'+seed+'_recon.npy')

    if method=='dca' or method=='deepimpute':
        l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, rmse = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
        cosine = imputation_cosine(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)    
    else:
        l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, rmse = imputation_error_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
        cosine = imputation_cosine_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, cosine, rmse))


for method in methodList:
    for datasetName in datasetList:
        for seed in seedList:
            for ratio in ratioList:        
                outResults(datasetName=datasetName, seed=seed, ratio=ratio, method=method)