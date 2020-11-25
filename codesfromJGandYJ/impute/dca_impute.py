import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import csv
import argparse
import sys

parser = argparse.ArgumentParser(description='Imputation DCA')
# In this script, not using arguments
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1', help='dropoutratio')
args = parser.parse_args()


def impute_dca(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)
    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)

    save_path = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/dca/{}/'.format(args.data)

    features=x.T

    #write
    dropout_filename = save_path+datasetNameStr+"_dropout.csv"
    with open(dropout_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    os.system("dca "+dropout_filename+ " "+save_path+datasetNameStr)

    filename=save_path+datasetNameStr+"/mean.tsv"
    imputed_values = pd.read_csv(filename,sep="\t")
    imputed_values=imputed_values.T

    np.save('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/dca/{}/{}_{}_recon.npy'.format(args.data,datasetNameStr,args.ratio),imputed_values)