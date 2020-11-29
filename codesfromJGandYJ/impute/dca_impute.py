import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import sys

# Ref: https://github.com/theislab/dca
# Notes: As tensorflow comes to 2.0 version, lots of things chagned, here is the version tested in Nov.26, 2020
# python==3.7.9
# tensorflow==1.15.4
# keras==2.3.1
# theano==1.0.5
# scanpy==1.5.1

parser = argparse.ArgumentParser(description='Imputation DCA')
parser.add_argument('--origin', action='store_true', default=False, help='Whether use origin (default: use ratio 0.0)')
args = parser.parse_args()

save_path = '/storage/htc/joshilab/wangjue/scGNN/tmp/'

def impute_dca(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)
    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    features=x.T

    #write
    dropout_filename = save_path+"dca_input.csv"
    with open(dropout_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    os.system("dca "+dropout_filename+ " "+save_path+"tmpdca")

    filename=save_path+"tmpdca/mean.tsv"
    imputed_values = pd.read_csv(filename,sep="\t")
    imputed_values=imputed_values.T

    np.save('/storage/htc/joshilab/wangjue/scGNN/dca/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),imputed_values)

datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

if args.origin:
    for datasetName in datasetNameList:
        impute_dca(seed='1', datasetName=datasetName, ratio='0.0')
else:
    for datasetName in datasetNameList:
        for seed in seedList:
            for ratio in ratioList:        
                impute_dca(seed=seed, datasetName=datasetName, ratio=ratio)
