import numpy as np
import pandas as pd
import argparse
import scipy.sparse
import sys
import magic
sys.path.append('../')
from util_function import *
from benchmark_util import *
from sklearn.cluster import KMeans
from sklearn.metrics import *
import matplotlib.pyplot as plt 

#Evaluating imputing results
#Used to postprocess results of imputation
parser = argparse.ArgumentParser(description='Imputation Results')
parser.add_argument('--datasetName', type=str, default='12.Klein',
                    help='databaseName')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
# if have benchmark: use cell File
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/allBench/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv',help="label Filename")
args = parser.parse_args()

labelFilename = args.labelFilename
true_labels = readTrueLabelList(labelFilename)

npyDir = '/storage/htc/joshilab/wangjue/scGNN/'

featuresOriginal = load_data(args.datasetName, False)
features         = None
dropi            = np.load(npyDir+'npyImputeG2E_1/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropi.npy')
dropj            = np.load(npyDir+'npyImputeG2E_1/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropj.npy')
dropix           = np.load(npyDir+'npyImputeG2E_1/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_dropix.npy')

# for MAGIC
fO = featuresOriginal.todense()
oriz = fO.copy()
for item in dropix:
	oriz[dropi[item],dropj[item]]=0.0

# Add log transformation
x = np.log(oriz+1)

# Load single-cell RNA-seq data
# Default is KNN=5
magic_operator = magic.MAGIC()
# magic_operator = magic.MAGIC(knn=10)
X_magic = magic_operator.fit_transform(x, genes="all_genes")
recon_magic = X_magic

def findoverlap(A,B):
    '''
    A: np.nonzero() 
    B:
    '''
    num = 0
    AA=[]
    BB=[]
    for i in range(len(A[0])):
        AA.append(str(A[0][i])+'-'+str(A[1][i]))
    for i in range(len(B[0])):
        BB.append(str(B[0][i])+'-'+str(B[1][i]))
    CC=set(AA).intersection(BB)
    return CC

def getAllResults(featuresImpute,featuresOriginal):
    #original
    l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)


    A = np.nonzero(featuresImpute)
    fO = featuresOriginal.todense()
    fO = np.asarray(fO)
    B = np.nonzero(fO)

    P = len(B[0])
    N = fO.shape[0]*fO.shape[1] - P
    Ppredict = len(A[0])
    Npredict = fO.shape[0]*fO.shape[1] - Ppredict

    CC=findoverlap(A,B)

    TP = len(CC)
    FN = P - TP
    FP = Ppredict - TP
    TN = Npredict - FN

    TPR = TP/P #sensitivity
    TNR = TN/N #specificity
    PPV = TP/(TP+FP) #precision
    # NPV = TN/(TN+FN) #negative predictive value Not used here for TN+FN may be 0
    FNR = FN/(FN+TP) #false negative rate
    FPR = FP/(FP+TN) #false positive rate
    FDR = FP/(FP+TP) 
    ACC = (TP+TN)/(P+N)
    F1  = 2*TP/(2*TP+FP+FN)
    MCC = (TP*TN-FP*FN)/np.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

    BB = fO.reshape(fO.shape[0]*fO.shape[1],)
    AA = featuresImpute.reshape(fO.shape[0]*fO.shape[1],)
    results = np.where(BB > 0, 1, 0)
    # calculate scores
    # Method 1: exp
    # scale = lambda x: 1-np.exp(-x)
    # Method 2: linear
    # scale = lambda x: (x-np.min(featuresImpute))/(np.max(featuresImpute)-np.min(featuresImpute))
    # Method 3: linear + log
    scale = lambda x: np.log((x-np.min(featuresImpute))/(np.max(featuresImpute)-np.min(featuresImpute))+1)
    scores = scale(AA)

    AUC=roc_auc_score(results, scores)

    # print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:d} {:d} {:d} {:d} {:.4f} {:.4f} {:.4f} {:.4f} '.format(F1,MCC,AUC,TPR,TNR,PPV,NPV,FNR,FPR,FDR,ACC,TP,FN,FP,TN,l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:d} {:d} {:d} {:d} {:.4f} {:.4f} {:.4f} {:.4f} '.format(F1,MCC,AUC,TPR,TNR,PPV,FNR,FPR,FDR,ACC,TP,FN,FP,TN,l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')   
    print('')

    fpr_, tpr_, _ = roc_curve(results, scores)
    precision_, recall_, _ = precision_recall_curve(results, scores)

    return fpr_, tpr_, precision_, recall_

featuresImpute1 = np.load(npyDir+'npyImputeG2E_1/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute2 = np.load(npyDir+'npyImputeG2E_2/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute3 = np.load(npyDir+'npyImputeG2E_3/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute4 = np.load(npyDir+'npyImputeG2E_4/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute5 = np.load(npyDir+'npyImputeG2E_5/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute6 = np.load(npyDir+'npyImputeG2E_6/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute7 = np.load(npyDir+'npyImputeG2E_7/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute8 = np.load(npyDir+'npyImputeG2E_8/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')
featuresImpute9 = np.load(npyDir+'npyImputeG2E_9/'+args.datasetName+'_LTMG_'+args.ratio+'_10-0.1-0.9-0.0-0.3-0.1_recon.npy')

fpr_m, tpr_m, precision_m, recall_m = getAllResults(recon_magic,featuresOriginal)
fpr_1, tpr_1, precision_1, recall_1 = getAllResults(featuresImpute1,featuresOriginal)
fpr_2, tpr_2, precision_2, recall_2 = getAllResults(featuresImpute2,featuresOriginal)
fpr_3, tpr_3, precision_3, recall_3 = getAllResults(featuresImpute3,featuresOriginal)
fpr_4, tpr_4, precision_4, recall_4 = getAllResults(featuresImpute4,featuresOriginal)
fpr_5, tpr_5, precision_5, recall_5 = getAllResults(featuresImpute5,featuresOriginal)
fpr_6, tpr_6, precision_6, recall_6 = getAllResults(featuresImpute6,featuresOriginal)
fpr_7, tpr_7, precision_7, recall_7 = getAllResults(featuresImpute7,featuresOriginal)
fpr_8, tpr_8, precision_8, recall_8 = getAllResults(featuresImpute8,featuresOriginal)
fpr_9, tpr_9, precision_9, recall_9 = getAllResults(featuresImpute9,featuresOriginal)


plt.figure()
plt.plot(fpr_m, tpr_m, ':k', label='MAGIC')
# plt.plot(fpr_6, tpr_6, 'r', label='L1-0.1')
# plt.plot(fpr_7, tpr_7, 'b', label='L2-0.1')
plt.plot(fpr_8, tpr_8, 'r', label='L1:1.0')
plt.plot(fpr_9, tpr_9, 'b', label='L2:1.0')
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('Figure-'+args.datasetName+'-'+args.ratio+'-ROC.eps', dpi=300)

plt.figure()
plt.plot(precision_m, recall_m, ':k', label='MAGIC')
# plt.plot(precision_6, recall_6, 'r', label='L1-0.1')
# plt.plot(precision_7, recall_7, 'b', label='L2-0.1')
plt.plot(precision_8, recall_8, 'r', label='L1:1.0')
plt.plot(precision_9, recall_9, 'b', label='L2:1.0')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.savefig('Figure-'+args.datasetName+'-'+args.ratio+'-PrecisionRecall.eps', dpi=300)



