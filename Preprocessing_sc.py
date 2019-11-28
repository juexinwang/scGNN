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
parser.add_argument('--expression-name', type=str, default='5.Pollen',
                    help='TGFb from MAGIC/ test also from MAGIC/ sci-CAR/ sci-CAR_LTMG/ 5.Pollen')

args = parser.parse_args()

def preprocess_network(feature_filename):
    '''
    Preprocessing by read expression
    Now it outputs cells and genes not zero
    output geneList, geneDict    
    '''
    # geneList, geneDict
    geneList=[]
    geneDict={}

    # Check cell and genes
    count = -1
    exDict={}
    exReadDict={}
    cellTag = False
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
                    cellReadCount = cellReadCount + float(word)
                    if tcount in exReadDict:
                        exReadDict[tcount] = exReadDict[tcount] + float(word)
                    else:
                        exReadDict[tcount] = 0.0
                    tcount = tcount + 1
                if cellReadCount == 0.0:
                    print("Cell "+str(count)+" has 0 reads") 
                    cellTag = True
            count = count+1
    f.close()

    if cellTag:
        print("All Cells are included.")

    for index in exReadDict:
        gene = exDict[index]
        if exReadDict[index] != 0.0:
            geneList.append(gene)
            geneDict[gene] = index
        # Debug usage
        # else:
        #     print("Gene "+str(index)+": "+gene+" has 0 reads")

    return geneList, geneDict

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
                print(str(ytcount)+"\t"+str(ntcount))
            if count >= 0:
                #discrete here
                tmplist =[]
                for word in words:
                    tmplist.append(float(word))
                avgtmp = np.sum(tmplist)/float(len(tmplist))
                
                data_count = 0
                for item in selectList:
                    samplelist.append(count)
                    featurelist.append(data_count)
                    # if discrete_tag == 'Avg':
                    if tmplist[item]>=avgtmp:
                        dataD.append(1)
                    else:
                        dataD.append(0)
                    # elif discrete_tag == 'Ori':
                    data.append(float(tmplist[item]))
                    data_count += 1
            count += 1
    f.close()
    # As dream: rows as cells, columns as genes: This is transpose of the original scRNA data
    feature = scipy.sparse.csr_matrix((data, (samplelist, featurelist)), shape=(count,len(selectList))) 
    featureD = scipy.sparse.csr_matrix((dataD, (samplelist, featurelist)), shape=(count,len(selectList))) 

    # For Matlab
    dim2out = [[0.0] * len(selectList)  for i in range(count)]
    dim2outD = [[0.0] * len(selectList) for i in range(count)]
    
    count = -1
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
                    dim2out[count][data_count]=float(tmplist[item])
                    if tmplist[item]>=avgtmp:
                        dim2outD[count][data_count]=1
                    else:
                        dim2outD[count][data_count]=0
                    data_count += 1
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
elif args.expression_name=='2.Yan':
    expressionname = '2.Yan.csv'
elif args.expression_name=='5.Pollen':
    expressionname = '5.Pollen.csv'
elif args.expression_name=='test':
    expressionname = 'test_data.csv'

out_folder = "data/sc/"+args.expression_name+"/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

feature_filename = "/home/wangjue/biodata/scData/"+expressionname

geneList, geneDict = preprocess_network(feature_filename)

#python and matlab
#First generate feature
feature, featureD, dim2out, dim2outD = read_feature_file_sparse(feature_filename, geneList, geneDict)

# Try to generate the graph structure
# edgeList = cal_distanceMatrix(feature, k=5)
# graphcsc, rowO, colO, dataO  = read_edge_file_csc(edgeList, feature.shape[0], k=5)
# graphdict = read_edge_file_dict(edgeList, feature.shape[0] )
outname = args.expression_name

x = feature
tx = feature[0:100]
allx = feature[100:]
# Discrete
xD = featureD
txD = featureD[0:100]
allxD = featureD[100:]

testindex = ""
for i in range(100):
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

