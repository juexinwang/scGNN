import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature
import scipy

recon = np.load('recon.npy')
# VIM/CDH1/ZEB1 129/68/833
# for sci-CAR
# 128/72/822
gene1 = 128
gene2 = 72
gene3 = 822

plt.scatter(recon[:,gene1], recon[:,gene2], c=recon[:,gene3], cmap="inferno")
plt.xlabel('VIM')
plt.ylabel('CDH1')
plt.legend('ZEB1')
# show the plot
# plt.show()
plt.savefig('result.png')

scipy.stats.pearsonr(recon[:,gene1], recon[:,gene2])
scipy.stats.pearsonr(recon[:,gene1], recon[:,gene3])
scipy.stats.pearsonr(recon[:,gene2], recon[:,gene3])
scipy.stats.spearmanr(recon[:,gene1], recon[:,gene2])
scipy.stats.spearmanr(recon[:,gene1], recon[:,gene3])
scipy.stats.spearmanr(recon[:,gene2], recon[:,gene3])



# Original
plt.scatter(recon[:,0,129], recon[:,0,68], c=recon[:,0,833], cmap="inferno")
plt.xlabel('VIM')
plt.ylabel('CDH1')
plt.legend('ZEB1')
scipy.stats.pearsonr(recon[:,0,129], recon[:,0,68])
scipy.stats.pearsonr(recon[:,0,129], recon[:,0,833])
scipy.stats.pearsonr(recon[:,0,68], recon[:,0,833])
scipy.stats.spearmanr(recon[:,0,129], recon[:,0,68])
scipy.stats.spearmanr(recon[:,0,129], recon[:,0,833])
scipy.stats.spearmanr(recon[:,0,68], recon[:,0,833])


#cell
plt.scatter(recon[129,:], recon[68,:], c=recon[833,:], cmap="inferno")
plt.xlabel('VIM')
plt.ylabel('CDH1')
plt.show()

scipy.stats.pearsonr(recon[129,:], recon[68,:])
scipy.stats.pearsonr(recon[129,:], recon[833,:])
scipy.stats.pearsonr(recon[68,:], recon[833,:])
scipy.stats.spearmanr(recon[129,:], recon[68,:])
scipy.stats.spearmanr(recon[129,:], recon[833,:])
scipy.stats.spearmanr(recon[68,:], recon[833,:])
