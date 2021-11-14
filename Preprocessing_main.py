import os
import time
import numpy as np
import argparse
from copy import deepcopy
from scipy import interpolate
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
import scipy.sparse
import sys
import pickle
import csv

# Preprocess network for sc
parser = argparse.ArgumentParser()
parser.add_argument('--expression-name', type=str, default='Chung',
                    help='TGFb from MAGIC/test also from 9.Chung/11.Kolodziejczyk/12.Klein/13.Zeisel')
parser.add_argument('--featureDir', type=str, default='/Users/wangjue/workspace/scGNN/',
                    help='Feature File Directory')
parser.add_argument('--data-type', type=str, default='float',
                    help='int/float')
parser.add_argument('--geneNzThreshold', type=float, default=0.0,
                    help='cells with genes not zero at least (default: 0.05) Choose 0.0 to let it as is')  
parser.add_argument('--geneThreshold', type=int, default=2000,
                    help='how many genes are selected (default: 2000)')
parser.add_argument('--countThreshold', action='store_true', default=False,
                    help='use count as the threshold')
parser.add_argument('--cell-threshold', type=int, default=-1,
                    help='1000 for varID, -1 for all')
parser.add_argument('--gene-threshold', type=int, default=-1,
                    help='1000 for varID, -1 for all')                  
                    
args = parser.parse_args()

if args.data_type == 'int':
    zero = 0
elif args.data_type == 'float':
    zero = 0.0

# Old, threshold 1000 as VarID
def preprocess_network_countsThreshold(feature_filename, cellthreshold=1000, genethreshold=1000):
    '''
    Preprocessing by read expression by counts threshold
    Now it outputs cells and genes larger than threshold
    output geneList, geneDict, cellList, cellDict    
    '''
    # geneList, geneDict
    geneList=[]
    geneDict={}
    cellList=[]
    cellDict={}

    # Check cell and genes
    count = -1
    exDict={}
    exReadDict={}
    with open(feature_filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            words = line.split(',')
            if count == -1:
                tcount =0
                for word in words:
                    exDict[tcount] = word
                    tcount = tcount + 1
            else:
                cellReadCount = 0
                tcount = 0
                for word in words:                    
                    if tcount in exReadDict:
                        exReadDict[tcount] = exReadDict[tcount] + float(word)
                        cellReadCount = cellReadCount + float(word)
                    else:
                        exReadDict[tcount] = zero
                    tcount = tcount + 1
                if cellReadCount < cellthreshold:
                    print("Cell "+str(count)+" has less than "+ str(cellthreshold) +" reads")
                else:
                    cellList.append(count)
                    cellDict[count]=''
            count = count+1
    f.close()

    for index in exReadDict:
        gene = exDict[index]
        if exReadDict[index] >= genethreshold:
            geneList.append(gene)
            geneDict[gene] = index
        # Debug usage
        # else:
        #     print("Gene "+str(index)+": "+gene+" has 0 reads")

    return geneList, geneDict, cellList, cellDict

# Prefer to use threshold here
def preprocess_network(feature_filename, geneNzThreshold=0.05, geneThreshold=2000):
    '''
    Preprocessing by read expression
    Now it outputs all cells and genes nonzero than threshold of all cells
    output geneList, geneDict, cellList, cellDict    
    '''
    # geneList, geneDict
    geneList=[]
    geneDict={}
    cellList=[]
    cellDict={}
    exDict={}

    #TODO: create a huge matrix, can be update later
    # Get cell number and gene number
    count = -1
    with open(feature_filename) as f:
        lines = f.readlines()
        for line in lines:
            if count == -1:
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                words = line.split(',')
                tcount =0
                for word in words:
                    exDict[tcount] = word
                    tcount = tcount + 1
            else:
                cellList.append(count)
                cellDict[count]=''
            count = count + 1
    f.close()
    
    cellcount = count
    genecount = tcount

    genenzThresholdCount = (int)(cellcount * geneNzThreshold)
    # cell as the rows, gene as the col 
    contentArray = [[0.0] * genecount for i in range(cellcount)]

    # gene nonezero count List
    genenzCountList=[0] * genecount

    # Check cell and genes
    count = -1
    with open(feature_filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            words = line.split(',')
            if count > -1:
                tcount = 0
                for word in words:
                    contentArray[count][tcount] = float(word)
                    if not float(word)==0.0:
                        genenzCountList[tcount]=genenzCountList[tcount]+1 
                    tcount = tcount + 1
            count = count+1
            if count%1000 == 0:
                print('{} of {} cells have been proceeded.'.format(count,cellcount))
    f.close()

    tmpindexList=[]
    for i in range(genecount):
        if genenzCountList[i]>genenzThresholdCount:
            tmpindexList.append(i)

    contentArray = np.asarray(contentArray)
    tmpindexList = np.asarray(tmpindexList)

    tmpChooseIndex = np.argsort(-np.var(contentArray[:,tmpindexList], axis=0))[:geneThreshold]
    tmpChooseIndex = tmpChooseIndex.tolist()
    chooseIndex = tmpindexList[tmpChooseIndex]

    for i in chooseIndex:
        gene = exDict[i]
        geneList.append(gene)
        geneDict[gene] = i

    return geneList, geneDict, cellList, cellDict

# For node as cell
# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, geneList, geneDict):
    samplelist=[]
    featurelist=[]
    data =[]
    dataD = []
    selectDict={}
    selectList=[]
    count = -1

    with open(filename) as f:
        lines = f.readlines()
        cellcount = 0
        for line in lines:            
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            words = line.split(',')
            if count == -1:
                tcount =0
                for word in words:
                    if word in geneDict:
                        selectDict[word] = tcount
                    tcount = tcount + 1
                ntcount = 0
                ytcount = 0
                for gene in geneList:
                    if gene in selectDict:
                        selectList.append(selectDict[gene])
                        ytcount += 1
                    else:
                        print(str(gene)+' is not in the input')
                        ntcount += 1
                # print(str(ytcount)+"\t"+str(ntcount))
            if count >= 0:
                #discrete here
                tmplist =[]
                for word in words:
                    tmplist.append(float(word))
                avgtmp = np.sum(tmplist)/float(len(tmplist))
                
                data_count = 0
                for item in selectList:
                    samplelist.append(cellcount)
                    featurelist.append(data_count)
                    # if discrete_tag == 'Avg':
                    if tmplist[item]>=avgtmp:
                        dataD.append(1)
                    else:
                        dataD.append(0)
                    # elif discrete_tag == 'Ori':
                    data.append(float(tmplist[item]))
                    data_count += 1
                cellcount += 1
            count += 1
    f.close()
    # As dream: rows as cells, columns as genes: This is transpose of the original scRNA data
    feature = scipy.sparse.csr_matrix((data, (samplelist, featurelist)), shape=(cellcount,len(selectList))) 
    featureD = scipy.sparse.csr_matrix((dataD, (samplelist, featurelist)), shape=(cellcount,len(selectList))) 

    # For Matlab
    dim2out = [[zero] * len(selectList)  for i in range(cellcount)]
    dim2outD = [[zero] * len(selectList) for i in range(cellcount)]
    
    count = -1
    cellcount = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            words = line.split(',')
            if count >= 0:
                tmplist =[]
                for word in words:
                    tmplist.append(float(word))
                avgtmp = np.sum(tmplist)/float(len(tmplist))
                
                data_count = 0
                for item in selectList:
                    dim2out[cellcount][data_count]=float(tmplist[item])
                    if tmplist[item]>=avgtmp:
                        dim2outD[cellcount][data_count]=1
                    else:
                        dim2outD[cellcount][data_count]=0
                    data_count += 1
                
                cellcount +=1
            count += 1
    f.close()

    return feature, featureD, dim2out, dim2outD

expressionname=args.expression_name

if args.expression_name=='TGFb':
    expressionname = 'HMLE_TGFb_day_8_10.csv'
    # expressionname = 'HMLE_TGFb_day_8_10_part.csv'
elif args.expression_name=='sci-CAR':
    expressionname = 'sci-CAR.csv'
elif args.expression_name=='sci-CAR_LTMG':
    expressionname = 'sci-CAR_LTMG.csv'
elif args.expression_name=='MMPbasal':
    expressionname = 'MMPbasal.csv'
elif args.expression_name=='MMPbasal_all':
    expressionname = 'MMPbasal.csv'
elif args.expression_name=='MMPbasal_allgene':
    expressionname = 'MMPbasal.csv'
elif args.expression_name=='MMPbasal_allcell':
    expressionname = 'MMPbasal.csv'
elif args.expression_name=='MMPbasal_2000':
    expressionname = 'MMPbasal.csv'
elif args.expression_name=='MMPbasal_2000_LTMG':
    expressionname = 'MMPbasal_2000_LTMG.csv'
elif args.expression_name=='MMPbasal_LTMG':
    expressionname = 'MMPbasal_LTMG.csv'
elif args.expression_name=='MMPbasal_all_LTMG':
    expressionname = 'MMPbasal_all_LTMG.csv'
elif args.expression_name=='MMPepo':
    expressionname = 'MMPepo.csv'
elif args.expression_name=='MMPepo_all':
    expressionname = 'MMPepo.csv'
elif args.expression_name=='MMPepo_allgene':
    expressionname = 'MMPepo.csv'
elif args.expression_name=='MMPepo_allcell':
    expressionname = 'MMPepo.csv'
elif args.expression_name=='T1000_LTMG':
    expressionname = 'T1000_LTMG.csv'
elif args.expression_name=='T2000_LTMG':
    expressionname = 'T2000_LTMG.csv'
elif args.expression_name=='T4000_LTMG':
    expressionname = 'T4000_LTMG.csv'
elif args.expression_name=='T8000_LTMG':
    expressionname = 'T8000_LTMG.csv'
elif args.expression_name=='T1000':
    expressionname = 'T1000.csv'
elif args.expression_name=='T2000':
    expressionname = 'T2000.csv'
elif args.expression_name=='T4000':
    expressionname = 'T4000.csv'
elif args.expression_name=='T8000':
    expressionname = 'T8000.csv'
elif args.expression_name=='test':
    expressionname = 'test_data.csv'
# Start working here
elif args.expression_name=='1.Biase':
    expressionname = '1.Biase.csv'
elif args.expression_name=='2.Li':
    expressionname = '2.Li.csv'
elif args.expression_name=='3.Treutlein':
    expressionname = '3.Treutlein.csv'
elif args.expression_name=='4.Yan':
    expressionname = '4.Yan.csv'
elif args.expression_name=='5.Goolam':
    expressionname = '5.Goolam.csv'
elif args.expression_name=='6.Guo':
    expressionname = '6.Guo.csv'
elif args.expression_name=='7.Deng':
    expressionname = '7.Deng.csv'
elif args.expression_name=='8.Pollen':
    expressionname = '8.Pollen.csv'
elif args.expression_name=='9.Chung':
    expressionname = '9.Chung.csv'
elif args.expression_name=='10.Usoskin':
    expressionname = '10.Usoskin.csv'
elif args.expression_name=='11.Kolodziejczyk':
    expressionname = '11.Kolodziejczyk.csv'
elif args.expression_name=='12.Klein':
    expressionname = '12.Klein.csv'
elif args.expression_name=='13.Zeisel':
    expressionname = '13.Zeisel.csv'
elif args.expression_name=='20.10X_2700_seurat':
    expressionname = '20.10X_2700_seurat.csv'
elif args.expression_name=='30.Schafer':
    expressionname = '30.Schafer.csv'
else:
    expressionname = args.expression_name+'.csv'

out_folder = "data/sc/"+args.expression_name+"/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

feature_filename = args.featureDir + expressionname

if args.countThreshold:
    #Set counts threshold as VarID
    geneList, geneDict, cellList, cellDict = preprocess_network_countsThreshold(feature_filename, cellthreshold=args.cell_threshold, genethreshold=args.gene_threshold)
else:
    #Set threshold
    geneList, geneDict, cellList, cellDict = preprocess_network(feature_filename, geneNzThreshold=args.geneNzThreshold, geneThreshold=args.geneThreshold)

#python and matlab
#First generate feature
feature, featureD, dim2out, dim2outD = read_feature_file_sparse(feature_filename, geneList, geneDict)
print(str(len(cellList))+" cells are retained")

# Try to generate the graph structure
# edgeList = cal_distanceMatrix(feature, k=5)
# graphcsc, rowO, colO, dataO  = read_edge_file_csc(edgeList, feature.shape[0], k=5)
# graphdict = read_edge_file_dict(edgeList, feature.shape[0] )
outname = args.expression_name

x = feature
tx = feature[0:1]
allx = feature[1:]
# Discrete
xD = featureD
txD = featureD[0:1]
allxD = featureD[1:]

testindex = ""
for i in range(1):
    testindex = testindex + str(i) + "\n"

pickle.dump(allx, open( out_folder+"ind."+outname+".allx", "wb" ) )
pickle.dump(x, open( out_folder+"ind."+outname+".x", "wb" ) )
pickle.dump(tx, open( out_folder+"ind."+outname+".tx", "wb" ) )
# graph
# pickle.dump(graphcsc, open( out_folder+"ind."+outname+".csc", "wb" ) )
# pickle.dump(graphdict, open( out_folder+"ind."+outname+".graph", "wb" ) )

# Output discrete
pickle.dump(allxD, open( out_folder+"ind."+outname+".allxD", "wb" ) )
pickle.dump(xD, open( out_folder+"ind."+outname+".xD", "wb" ) )
pickle.dump(txD, open( out_folder+"ind."+outname+".txD", "wb" ) )

with open ( out_folder+"ind."+outname+".test.index", 'w') as fw:
    fw.writelines(testindex)
    fw.close()

# For matlab
with open(out_folder+outname+'.features.csv','w') as fw:
    writer = csv.writer(fw)
    writer.writerows(dim2out)
fw.close()

with open(out_folder+outname+'.features.D.csv','w') as fw:
    writer = csv.writer(fw)
    writer.writerows(dim2outD)
fw.close()

# with open(out_folder+outname+'.row.csv','w') as fw:
#     for item in rowO:
#         fw.write(str(item)+"\n")
# fw.close()

# with open(out_folder+outname+'.col.csv','w') as fw:
#     for item in colO:
#         fw.write(str(item)+"\n")
# fw.close()

# with open(out_folder+outname+'.data.csv','w') as fw:
#     for item in dataO:
#         fw.write(str(item)+"\n")
# fw.close()


# gene name:
with open(out_folder+outname+'.gene.txt','w') as fw:
    for gene in geneList:
        fw.write(gene+"\n")
fw.close()

