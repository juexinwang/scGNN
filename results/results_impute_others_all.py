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
# In HPC, call by sbatch submit_Impute_others.sh

datasetList = [
    '9.Chung',
    '11.Kolodziejczyk',
    '12.Klein',
    '13.Zeisel',
]

oridirStr = '../npyImputeG2E'
medirStr = '../'

seedList = ['1','2','3']
ratioList = ['0.1','0.3','0.6','0.8']

# sophisticated, not using
# methodList = ['magic','saucie','saver','scimpute','scvi','scvinorm','dca','deepimpute','scIGANslog','scIGANs','netNMFsclog','netNMFsc']

# We should use only log(x+1) if the method permitted
# methodList = ['magic','saucie','saver','scimpute','scvi','scvinorm','dca','deepimpute','scIGANs','netNMFsc']

# Temp: just test dca
methodList = ['dca']

def outResults(datasetName,seed,ratio,method):
    featuresOriginal = load_data(datasetName, discreteTag=False)

    features         = None
    dropi            = np.load(oridirStr+'_'+seed+'/'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropi.npy')
    dropj            = np.load(oridirStr+'_'+seed+'/'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropj.npy')
    dropix           = np.load(oridirStr+'_'+seed+'/'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropix.npy')

    # scGNN results
    # featuresImpute   = np.load(npyDir+datasetName+'_'+args.regulized_type+discreteStr+'_'+args.ratio+'_10-0.1-0.9-0.0-0.3-'+args.regupara+'_recon'+args.reconstr+'.npy')
    if method == 'scvinorm':
        featuresImpute   = np.load(medirStr+'scvi/'+datasetName+'_'+ratio+'_'+seed+'_recon_normalized.npy')
    # not using now
    elif method == 'scIGANs':
        df = pd.read_csv('/storage/htc/joshilab/jghhd/singlecellTest/scIGAN/Result_200_'+ratio+'/'+datasetName+'/scIGANs_npyImputeG2E_'+seed+'_'+datasetName+'_LTMG_'+ratio+'_10-0.1-0.9-0.0-0.3-0.1_features_log.csv_'+datasetName.split('.')[1]+'_only_label.csv.txt',sep='\s+',index_col=0)
        tmp = df.to_numpy()
        featuresImpute   = tmp.T
    elif method == 'netNMFsc':
        featuresImpute   = np.load('/storage/htc/joshilab/jghhd/singlecellTest/netNMFsc/result_mi_100000/'+ratio+'/'+datasetName+'/npyImputeG2E_'+seed+'_log_imputation.npy')
        featuresImpute = featuresImpute.T
    else:
        featuresImpute   = np.load(medirStr+method+'/'+datasetName+'_'+ratio+'_'+seed+'_recon.npy')

    # No log
    if method=='dca' or method=='deepimpute':
        l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, rmse = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
        cosine = imputation_cosine(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)    
    # log
    else:
        l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, rmse = imputation_error_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
        cosine = imputation_cosine_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, cosine, rmse))


for method in methodList:
    for datasetName in datasetList:
        for seed in seedList:
            for ratio in ratioList:        
                outResults(datasetName=datasetName, seed=seed, ratio=ratio, method=method)