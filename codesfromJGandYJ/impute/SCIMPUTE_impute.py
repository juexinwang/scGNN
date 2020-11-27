import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import sys

# Notes in install scimpute:
# Have to add in R: 
# Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS=TRUE)
# Ref: https://github.com/Vivianstats/scImpute

parser = argparse.ArgumentParser(description='Impute scImpute')
# In this script, not using arguments
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1', help='dropoutratio')
args = parser.parse_args()

save_path = '/storage/htc/joshilab/wangjue/scGNN/tmp/'

def impute_scimpute(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)

    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    x=np.log(x+1)

    features = np.copy(x)

    #transpose and add names for rows and cols
    features=np.transpose(features)
    rowname=np.linspace(1,features.shape[0],features.shape[0]).reshape([features.shape[0],1])
    features=np.concatenate([rowname,features],axis=1)
    colname=np.linspace(1,features.shape[1],features.shape[1]).reshape([1,features.shape[1]])
    features=np.concatenate([colname,features],axis=0)

    features=features.T

    #write
    dropout_filename = save_path+"scimpute_input.csv"
    with open(dropout_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    #run the R script
    os.system("Rscript scimpute.r "+save_path+"scimpute_input.csv "+save_path+"tmpscimpute/")

    filename=save_path+"tmpscimpute/scimpute_count.csv"
    imputed_values = pd.read_csv(filename,sep=",",index_col=0)
    imputed_values = imputed_values.to_numpy()

    np.save('/storage/htc/joshilab/wangjue/scGNN/scimpute/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),imputed_values)

datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

for datasetName in datasetNameList:
    for seed in seedList:
        for ratio in ratioList:        
            impute_scimpute(seed=seed, datasetName=datasetName, ratio=ratio)

