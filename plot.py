import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature

recon = np.load('recon.npy')
# VIM/CDH1/ZEB1 129/68/833
plt.scatter(recon[:,129], recon[:,68], c=recon[:,833], cmap="inferno")
plt.xlabel('VIM')
plt.ylabel('CDH1')
plt.legend('ZEB1')
# show the plot
# plt.show()
plt.savefig('result.png')