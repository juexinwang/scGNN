import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import sys

# Ref:
# https://github.com/mohuangx/SAVER
# https://mohuangx.github.io/SAVER/articles/saver-tutorial.html
# Use python to generate input for saver.r, then output

parser = argparse.ArgumentParser(description='Impute SAVER')
# In this script, not using arguments
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1', help='dropoutratio')
args = parser.parse_args()

save_path = '/storage/htc/joshilab/wangjue/scGNN/tmp/'

def impute_saver(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)

    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    x=np.log(x+1)
    features=x.T

    #write
    dropout_filename = save_path+"saver_input.csv"
    with open(dropout_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    #run the R script
    os.system("Rscript saver.r "+save_path+"saver_input.csv "+save_path+"saver_output.csv ")

    filename=save_path+"saver_output.csv"
    imputed_values = pd.read_csv(filename,sep="\t")
    imputed_values=imputed_values.T

    np.save('/storage/htc/joshilab/wangjue/scGNN/saver/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),imputed_values)

datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

for datasetName in datasetNameList:
    for seed in seedList:
        for ratio in ratioList:        
            impute_saver(seed=seed, datasetName=datasetName, ratio=ratio)




