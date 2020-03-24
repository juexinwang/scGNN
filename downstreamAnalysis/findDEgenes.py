import numpy as np
import pandas as pd
from scipy import stats


parser = argparse.ArgumentParser(description='Find DE genes of each cluster')
parser.add_argument('--celltypeFilename', type=str, default='/storage/htc/joshilab/wangjue/scGNN/casenpyG2E_LB_/AD_GSE103334_NORMED_8CT_LTMG_0.5_0.0_0.0_results.txt',
                    help='celltype Filename of the celltype')
parser.add_argument('--csvFilename', type=str, default='/storage/htc/joshilab/wangjue/casestudy/AD_GSE103334_NORMED_8CT/Use_expression.csv',
                    help='expression filename')
parser.add_argument('--outputFilename', type=str, default='out.csv',
                    help='outputFilename')
parser.add_argument('--threshold', type=float, default=0.8,
                    help='percentage of cells of a celltype')
parser.add_argument('--zthreshold', type=float, default=0.0,
                    help='zscore threshold of cells of a celltype')
args = parser.parse_args()

clusters = pd.read_csv(args.celltypeFilename)
clusterDict = {}
celltypecount = 0
for row in clusters.itertuples():
    if row[2] in clusterDict:
        tlist = clusterDict[row[2]]
        tlist.append(row[0])
        clusterDict[row[2]]=tlist
    else:
        clusterDict[row[2]]=[row[0]]
        celltypecount +=1

matrix = pd.read_csv(args.csvFilename, index_col=0)
genelist = matrix.index.tolist()
celllist = matrix.columns.values.tolist()
matrix = matrix.to_numpy()

zmatrix = stats.zscore(matrix,axis=1)

outmatrix = np.zeros((len(genelist),celltypecount))

for gene in range(len(genelist)):
    for celltype in range(celltypecount):
        tlist = clusterDict[celltype]
        count = 0
        for cell in range(len(tlist)):
            if zmatrix[gene,tlist[cell]] > args.zthreshold:
                count += 1
        if count > len(tlist)*args.threshold:
            outmatrix[gene,celltype] = 1
outmatrix=outmatrix.astype(int)        
out_df = pd.DataFrame(outmatrix,index=genelist)
out_df.to_csv(outputFilename)

a,b=np.where(outmatrix>0)



