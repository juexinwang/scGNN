import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepimpute.multinet import MultiNet
import torch
import csv
import argparse
import sys

parser = argparse.ArgumentParser(description='Impute Deepimpute')
# In this script, not using arguments
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1', help='dropoutratio')
args = parser.parse_args()

# Ref:
# https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/data_loading.ipynb
save_path = '/storage/htc/joshilab/wangjue/scGNN/tmp/'

def impute_deepimpute(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)
    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    # x=np.log(x+1)

    features=x
    dropout_filename = save_path+"deepimpute.csv"
    with open(dropout_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    data = pd.read_csv(dropout_filename, header=None)
    model = MultiNet()
    model.fit(data)
    imputed = model.predict(data)

    np.save('/storage/htc/joshilab/wangjue/scGNN/deepimpute/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),imputed)

datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

for datasetName in datasetNameList:
    for seed in seedList:
        for ratio in ratioList:        
            impute_deepimpute(seed=seed, datasetName=datasetName, ratio=ratio)
