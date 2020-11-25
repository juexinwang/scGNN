#from dca.api import dca
#import anndata
#import matplotlib.pyplot as plt
#import numpy as np
#import time
#import pandas as pd

#Ref:
# https://github.com/theislab/dca/blob/master/tutorial.ipynb
#z = pd.read_csv('/home/wangjue/biodata/scData/MMPbasal.csv')
#z = z.to_numpy()
#z = z[:,:-1]

#selected = np.std(z, axis=0).argsort()[-2000:][::-1]
#expression_data = z[:, selected]

#train = anndata.AnnData(expression_data)
#res = dca(train, verbose=True)
#train.X

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
parser.add_argument('--outfolder', type=str, default='/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/otherresults/dca/',
                    help='output filefolder')
args = parser.parse_args()


if args.discreteTag:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scData/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/{}/{}_LTMG_0.1_features.npy'.format(args.data,args.datasetName)
x = np.load(filename,allow_pickle=True)
x = x.tolist()
x=x.todense()
x=np.asarray(x)
filenameFull = filename
save_path = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/dca/{}/'.format(args.data)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr



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