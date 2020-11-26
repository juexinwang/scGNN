import sys
sys.path.append("/storage/htc/joshilab/wangjue/")
import SAUCIE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Impute use SAUCIE')
# In this script, not using arguments
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1', help='dropoutratio')
args = parser.parse_args()

# modified from official tutorial: https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/exercises/Deep_Learning/notebooks/02_Answers_Exploratory_analysis_of_single_cell_data_with_SAUCIE.ipynb
def impute_saucie(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)
    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    x=np.log(x+1)
    x=np.transpose(x)
    loader_train = SAUCIE.Loader(x, shuffle=True)
    loader_eval = SAUCIE.Loader(x, shuffle=False)
    # clear the computational graph
    tf.reset_default_graph()
    # build the SAUCIE model
    model = SAUCIE.SAUCIE(x.shape[1])
    # train the model!
    model.train(loader_train, steps=2000)
    reconstruction = model.get_reconstruction(loader_eval)
    reconstruction=np.transpose(reconstruction)
    np.save('/storage/htc/joshilab/wangjue/scGNN/saucie/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),reconstruction)

datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

for datasetName in datasetNameList:
    for seed in seedList:
        for ratio in ratioList:        
            impute_saucie(seed=seed, datasetName=datasetName, ratio=ratio)
