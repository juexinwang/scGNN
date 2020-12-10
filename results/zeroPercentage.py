#Calculate Zero percentage in each of the datasets
import numpy as np

def calcu(dataset='9.Chung',ratio=0.0):
    t=np.load('npyImputeG2E_1/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(dataset,ratio),allow_pickle=True)
    t=t.tolist()
    t=t.todense()
    zeroNum = np.where(t==0)[0].shape[0]
    allNum = t.shape[0]*t.shape[1]
    percent = zeroNum/allNum
    print('{} {} {}'.format(zeroNum,allNum,percent))

datasetList = [
    '9.Chung',
    '11.Kolodziejczyk',
    '12.Klein',
    '13.Zeisel',
]

ratioList = ['0.0','0.1','0.3','0.6','0.8']

for dataset in datasetList:
    for ratio in ratioList:
        calcu(dataset, ratio)