from dca.api import dca
import anndata
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

#Ref:
# https://github.com/theislab/dca/blob/master/tutorial.ipynb
z = pd.read_csv('/home/wangjue/biodata/scData/MMPbasal.csv')
z = z.to_numpy()
z = z[:,:-1]

selected = np.std(z, axis=0).argsort()[-2000:][::-1]
expression_data = z[:, selected]

train = anndata.AnnData(expression_data)
res = dca(train, verbose=True)
train.X

