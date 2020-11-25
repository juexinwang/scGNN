import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepimpute.multinet import MultiNet
import torch
import csv
import argparse
import sys
sys.path.append('../')
sys.path.append('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/')
from benchmark_util import impute_dropout


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='data1',help='data1,2,3')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--outfolder', type=str, default='/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/otherresults/deepimpute/',
                    help='output filefolder')
args = parser.parse_args()

# Ref:
# https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/data_loading.ipynb

if args.discreteTag:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scData/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/{}/{}_LTMG_0.1_features.npy'.format(args.data,args.datasetName)
x = np.load(filename,allow_pickle=True)
x = x.tolist()
x=x.todense()
x=np.asarray(x)
#x=np.log(x+1)
filenameFull = filename
save_path = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/deepimpute_nolog/{}/'.format(args.data)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

features=x
dropout_filename = save_path+datasetNameStr+"_dropout.csv"
with open(dropout_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerows(features)

data = pd.read_csv(dropout_filename, header=None)
model = MultiNet()
model.fit(data)
imputed = model.predict(data)


np.save(save_path+'{}_{}_recon.npy'.format(datasetNameStr,args.ratio),imputed)

