import numpy as np
from sklearn.metrics import *

# filename = '/Users/juexinwang/Downloads/benchmark/human_tissue_quick/huamn_atlas_20tissue_LTMG_0.9_0.0_0.0_results.txt'
filename = '/Users/juexinwang/Downloads/huamn_atlas_20tissue_LTMG_0.9_0.0_0.0_results.txt'
# filename = '/Users/juexinwang/Downloads/huamn_atlas_20tissue_LTMG_0.9_0.0_0.0_results_largrbatch.txt'



celltyper = []
celltyperDict = {}
celltypep = []
typed = 1
count = 0
with (open(filename)) as f:
    lines = f.readlines()
    for line in lines:
        if not count==0:
            line = line.strip()
            tmpA = line.split(',')
            celltypep.append(tmpA[1])
            typeR = tmpA[0].split('_')[1]        
            if not typeR in celltyperDict:
                celltyperDict[typeR] = typed
                celltyper.append(typed)
                typed += 1
            else:
                celltyper.append(celltyperDict[typeR])
        count +=1
    f.close()

# silhouette, chs, dbs = measureClusteringNoLabel(zOut, listResult)
ari = adjusted_rand_score(celltypep, celltyper)
ami = adjusted_mutual_info_score(celltypep, celltyper)
nmi = normalized_mutual_info_score(celltypep, celltyper)
cs  = completeness_score(celltypep, celltyper)
fms = fowlkes_mallows_score(celltypep, celltyper)
vms = v_measure_score(celltypep, celltyper)
hs  = homogeneity_score(celltypep, celltyper)
# ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(celltypep, celltyper)
# print(str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs))
print(str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs))
        
