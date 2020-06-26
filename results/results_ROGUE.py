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
# if have benchmark: use cell File
parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/allBench/12.Klein/Klein_cell_label.csv',help="label Filename")
args = parser.parse_args()

labelFilename = args.labelFilename
true_labels = readTrueLabelList(labelFilename)

npyDir = '/storage/htc/joshilab/wangjue/scGNN/'

featuresOriginal = load_data(args.datasetName, False)
oriz = featuresOriginal.todense()

# Add log transformation
x = np.log(oriz+1)

df = pd.DataFrame(x)
df.to_csv('12.data.csv',index=False,header=None)