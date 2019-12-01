import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature
import scipy
import pandas as pd
import matplotlib.cm as cm

#Original
df = pd.read_csv('/home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_cell_label.csv')
df.columns = ['Cell','Cluster']
z = np.load('5.Pollen_noreguD_z.npy')
plt.scatter(z[:,0],z[:,1],c=df['Cluster'],cmap=cm.brg)
plt.show()

