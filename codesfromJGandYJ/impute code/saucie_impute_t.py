import sys
import tensorflow as tf
sys.path.append('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/otherresults/SAUCIE-master/SAUCIE-master/')
from model import SAUCIE
from loader import Loader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
sys.path.append('../')
sys.path.append('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/')
from benchmark_util import impute_dropout

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='data1',help='data1,2,3')
parser.add_argument('--datasetName', type=str, default='MMPbasal',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False, 
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
args = parser.parse_args()

# x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)
if args.discreteTag:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scData/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/{}/{}_LTMG_0.1_features.npy'.format(args.data,args.datasetName)
x = np.load(filename,allow_pickle=True)
x = x.tolist()
x=x.todense()
x=np.asarray(x)
x=np.log(x+1)

x=np.transpose(x)

saucie = SAUCIE(x.shape[1])
loadtrain = Loader(x, shuffle=True)
saucie.train(loadtrain, steps=1000)

loadeval = Loader(x, shuffle=False)
reconstruction = saucie.get_reconstruction(loadeval)

reconstruction=np.transpose(reconstruction)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

# l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error(recon, featuresOriginal, None, dropi, dropj, dropix)
# print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

np.save('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/saucie_t/{}/{}_{}_recon.npy'.format(args.data,datasetNameStr,args.ratio),reconstruction)


