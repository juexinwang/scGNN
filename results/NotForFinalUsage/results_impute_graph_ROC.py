import numpy as np
import pandas as pd
import argparse
import scipy.sparse
import sys

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


fO = featuresOriginal.todense()
oriz = fO.copy()
for item in dropix:
	oriz[dropi[item],dropj[item]]=0.0

# Add log transformation
x = np.log(oriz+1)

# Load single-cell RNA-seq data
# MAGIC
def impute_MAGIC(x):
    import magic
    # Default is KNN=5
    magic_operator = magic.MAGIC()
    # magic_operator = magic.MAGIC(knn=10)
    X_magic = magic_operator.fit_transform(x, genes="all_genes")
    recon_magic = X_magic
    return recon_magic

# SAUCIE
def impute_SAUCIE(x):
    import SAUCIE
    x=np.transpose(x)
    saucie = SAUCIE.SAUCIE(x.shape[1])
    loadtrain = SAUCIE.Loader(x, shuffle=True)
    saucie.train(loadtrain, steps=1000)

    loadeval = SAUCIE.Loader(x, shuffle=False)
    reconstruction = saucie.get_reconstruction(loadeval)

    recon_saucie=np.transpose(reconstruction)
    return recon_saucie

# Deep Impute
def impute_deepimpute(oriz):
    from deepimpute.multinet import MultiNet
    sys.path.append('/storage/htc/joshilab/wangjue/software/')
    # have to use raw value
    data = pd.DataFrame.from_records(np.asarray(oriz))
    model = MultiNet()
    model.fit(data)
    recon_deepimpute = model.predict(data)
    recon_deepimpute = recon_deepimpute.to_numpy()
    return recon_deepimpute

# Read existed Results
def impute_read(fileName):
    recon_read = pd.read_csv(fileName,header=None)
    recon_read = recon_read.to_numpy()
    return recon_read

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

def getROCResults(featuresImpute,featuresOriginal):
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

#Confusion matrix related. 
# fpr_m, tpr_m, precision_m, recall_m = getROCResults(recon_magic,featuresOriginal)
# fpr_1, tpr_1, precision_1, recall_1 = gerROCResults(featuresImpute1,featuresOriginal)
# fpr_2, tpr_2, precision_2, recall_2 = gerROCResults(featuresImpute2,featuresOriginal)
# fpr_3, tpr_3, precision_3, recall_3 = gerROCResults(featuresImpute3,featuresOriginal)
# fpr_4, tpr_4, precision_4, recall_4 = gerROCResults(featuresImpute4,featuresOriginal)
# fpr_5, tpr_5, precision_5, recall_5 = gerROCResults(featuresImpute5,featuresOriginal)
# fpr_6, tpr_6, precision_6, recall_6 = gerROCResults(featuresImpute6,featuresOriginal)
# fpr_7, tpr_7, precision_7, recall_7 = gerROCResults(featuresImpute7,featuresOriginal)
# fpr_8, tpr_8, precision_8, recall_8 = gerROCResults(featuresImpute8,featuresOriginal)
# fpr_9, tpr_9, precision_9, recall_9 = gerROCResults(featuresImpute9,featuresOriginal)

# plt.figure()
# plt.plot(fpr_m, tpr_m, ':k', label='MAGIC')
# # plt.plot(fpr_6, tpr_6, 'r', label='L1-0.1')
# # plt.plot(fpr_7, tpr_7, 'b', label='L2-0.1')
# plt.plot(fpr_8, tpr_8, 'r', label='L1:1.0')
# plt.plot(fpr_9, tpr_9, 'b', label='L2:1.0')
# plt.xlabel('False Positive Rate') 
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.savefig('Figure-'+args.datasetName+'-'+args.ratio+'-ROC.eps', dpi=300)

# plt.figure()
# plt.plot(precision_m, recall_m, ':k', label='MAGIC')
# # plt.plot(precision_6, recall_6, 'r', label='L1-0.1')
# # plt.plot(precision_7, recall_7, 'b', label='L2-0.1')
# plt.plot(precision_8, recall_8, 'r', label='L1:1.0')
# plt.plot(precision_9, recall_9, 'b', label='L2:1.0')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc="lower right")
# plt.savefig('Figure-'+args.datasetName+'-'+args.ratio+'-PrecisionRecall.eps', dpi=300)

def getAllResultsL1CosLog(featuresImpute,featuresOriginal):
    #original
    l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
    cosine = imputation_cosine_log(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, cosine), end='')   
    print('')

def getAllResultsL1Cos(featuresImpute,featuresOriginal):
    #original
    l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
    cosine = imputation_cosine(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)
    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax, cosine), end='')   
    print('')


# recon_sauice = impute_SAUCIE(x)
# getAllResultsL1Cos(recon_sauice,featuresOriginal)

recon_deepimpute = impute_deepimpute(oriz)
getAllResultsL1CosLog(recon_deepimpute,featuresOriginal)

recon_magic = impute_MAGIC(x)
getAllResultsL1CosLog(recon_magic,featuresOriginal)

for i in range(1,4):
    recon_dca = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.dca.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_dca,featuresOriginal)

for i in range(1,4):
    recon_impute = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.deepimpute.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_impute,featuresOriginal)

for i in range(1,4):
    recon_magic = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.magic.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_magic,featuresOriginal)

for i in range(1,4):
    recon_saucie = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.saucie.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_saucie,featuresOriginal)

for i in range(1,4):
    recon_saver = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.saver.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_saver,featuresOriginal)

for i in range(1,4):
    recon_scimpute = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.scimpute.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_scimpute,featuresOriginal)

for i in range(1,4):
    recon_scvi = impute_read('/storage/htc/joshilab/wangjue/imputed/all/12.scvi.'+str(i)+'.csv')
    getAllResultsL1CosLog(recon_scvi,featuresOriginal)

getAllResultsL1CosLog(featuresImpute1,featuresOriginal)
getAllResultsL1CosLog(featuresImpute2,featuresOriginal)
getAllResultsL1CosLog(featuresImpute3,featuresOriginal)
getAllResultsL1CosLog(featuresImpute4,featuresOriginal)
getAllResultsL1CosLog(featuresImpute5,featuresOriginal)
getAllResultsL1CosLog(featuresImpute6,featuresOriginal)
getAllResultsL1CosLog(featuresImpute7,featuresOriginal)
getAllResultsL1CosLog(featuresImpute8,featuresOriginal)
getAllResultsL1CosLog(featuresImpute9,featuresOriginal)


