# This code has not cleaned yet
import sys,os
import numpy as np
import pandas as pd
import argparse
sys.path.append('../')
sys.path.append('/storage/htc/joshilab/jghhd/singlecellTest/scIGAN/scIGANs/')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--Randomdata', type=str, default='npyImputeG2E_1',help='npyImputeG2E_1,2,3')
parser.add_argument('--datasetName', type=str, default='12.Klein',help='12.Klein,13.Zeisel')
parser.add_argument('--process', type=str, default='null',help='log/null to process data')
parser.add_argument('--exec', type=str, default='scIGANs',help='12.Klein')
parser.add_argument('--dropratio', type=str, default='0.1',help='0.1，0.3，0.6，0.8')
parser.add_argument('--csvsavepath', type=str, default='/storage/htc/joshilab/jghhd/singlecellTest/Data/',help='12.Klein')
parser.add_argument('--labelpath', type=str, default='/storage/htc/joshilab/jghhd/singlecellTest/Data/',help='12.Klein')
parser.add_argument('--outpath', type=str, default='/storage/htc/joshilab/jghhd/singlecellTest/scIGAN/Result_200/',help='12.Klein')
parser.add_argument('--Epotch', type=str, default='200',help='epotch')
args = parser.parse_args()

# x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)

filename = '/storage/hpc/group/joshilab/scGNNdata/{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(args.Randomdata,args.datasetName,args.dropratio)
x = np.load(filename,allow_pickle=True)
x = x.tolist()
x=x.todense()
x=np.asarray(x)
if args.process=='log':
    x=np.log(x+1)
    saveintedir = '{}{}/{}_{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features_log.csv'.format(args.csvsavepath, args.datasetName,args.Randomdata,
                                                                                args.datasetName,args.dropratio)
elif args.process=='null':
    saveintedir = '{}{}/{}_{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.csv'.format(args.csvsavepath, args.datasetName,args.Randomdata,
                                                                                args.datasetName,args.dropratio)
#transpose and add names for rows and cols
features=np.transpose(x)

pd.DataFrame(features).to_csv(saveintedir,sep='\t')

label = '{}{}/{}_only_label.csv'.format(args.labelpath,args.datasetName,args.datasetName.split('.')[-1])
#/storage/htc/joshilab/jghhd/singlecellTest/Data/12.Klein/Klein_only_label.csv

cmd = '{} {} -l {} -e {} -o {}{}'.format(args.exec,saveintedir,label,args.Epotch,args.outpath,args.datasetName)
print(cmd)
os.system(cmd)
#scIGANs saveintedir -l  -e 50

# l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error(recon, featuresOriginal, None, dropi, dropj, dropix)
# print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

#np.save('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/saucie_t/{}/{}_{}_recon.npy'.format(args.data,datasetNameStr,args.ratio),reconstruction)
