import os
import numpy as np
import pandas as pd
import sys
import csv
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--outfolder', type=str, default='/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/otherresults/recon/round_2exp',
                    help='output filefolder')
args = parser.parse_args()

if args.discreteTag:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scData/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = "/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/otherresults/recon/recon_data2/{}_LTMG_0.1_0.0-0.3-0.1_recon.csv".format(args.datasetName)
filenameFull = filename
save_path = args.outfolder

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr


x = pd.read_csv(filename,header=None)
x=pow(2.71828,x)

features=np.transpose(x)
rowname=np.linspace(1,features.shape[0],features.shape[0]).reshape([features.shape[0],1])
features=np.concatenate([rowname,features],axis=1)
colname=np.linspace(1,features.shape[1],features.shape[1]).reshape([1,features.shape[1]])
features=np.concatenate([colname,features],axis=0)

#write
dropout_filename = save_path+datasetNameStr+".csv"
with open(dropout_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerows(features)



os.system("/storage/hpc/data/yjiang/too-many-cells/too-many-cells_0.2.2.0.sif make-tree --matrix-path "+dropout_filename+" --output "+save_path+">"+save_path+datasetNameStr+"_clusters.csv")

x = pd.read_csv(save_path+datasetNameStr+"_clusters.csv")
x=np.array(x)
x=x[:,1]
np.save(save_path+datasetNameStr+"_clusters.npy",x)
