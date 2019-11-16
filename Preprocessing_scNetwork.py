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
parser.add_argument('--graph-type', type=str, default='cell',
                    help='cell/gene, cell:cell as nodes in the graph, gene:gene as nodes in the graph')
parser.add_argument('--network-name', type=str, default='ttrust',
                    help='ttrust')
parser.add_argument('--expression-name', type=str, default='sci-CAR',
                    help='TGFb from MAGIC/ test also from MAGIC/ sci-CAR/ sci-CAR_D')
parser.add_argument('--discrete-flag', type=str, default='False',
                    help='True for average discrete')

args = parser.parse_args()

# Read network and expression
# output geneList, geneDict
def preprocess_network(edge_filename, feature_filename):

    # geneList, geneDict
    geneList=[]
    tList=[]
    tgeneDict={}
    geneDict={}

    genecount = 0
    with open(edge_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]

            # geneList, geneDict, tfDict
            if end1 not in tgeneDict:
                tgeneDict[end1] = genecount
                tList.append(end1)
                genecount = genecount + 1
            if end2 not in tgeneDict:
                tgeneDict[end2] = genecount
                tList.append(end2)
                genecount = genecount + 1
            # if end1 not in tfDict:
            #     tfDict[end1] = tgeneDict[end1]
    f.close()

    count = 0
    exDict={}
    with open(feature_filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            words = line.split(',')
            if count == 0:
                tcount =0
                for word in words:
                    exDict[word] = tcount
                    tcount = tcount + 1
            else:
                break
            count = count+1
    f.close()

    count = 0
    for gene in tList:
        if gene in exDict:
            geneList.append(gene)
            geneDict[gene] = count
            count +=1

    return geneList, geneDict

# Load gold standard edges into sparse matrix
# No edge types
# output mtx, tfDict
# Additional outfile for matlab
def read_edge_file_csc(filename, geneDict):
    row=[]
    col=[]
    data=[]

    tfDict={}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]
            
            #Add to sparse matrix
            if end1 in geneDict and end2 in geneDict:
                row.append(geneDict[end1])
                col.append(geneDict[end2])
                data.append(1.0)
                row.append(geneDict[end2])
                col.append(geneDict[end1])
                data.append(1.0)
                count += 1
                if end1 not in tfDict:
                    tfDict[end1] = geneDict[end1]
    f.close()
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(len(geneDict), len(geneDict)))
    
    #python output
    # return mtx, tfDict

    #Output for matlab
    return mtx, tfDict, row, col, data

class KNNEdge:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# Not use it now
# Calculate KNN graph, return row and col
def cal_distanceMatrix(featureMatrix, k=5):
    distMat = distance_matrix(featureMatrix.todense(),featureMatrix.todense())
    edgeList=[]

    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append(KNNEdge(i,res[j]))
    
    return edgeList


# For cell,use feature matrix as input, row as cells, col as genes
# Load gold standard edges into sparse matrix
# No edge types
# output mtx, tfDict
# Additional outfile for matlab
def read_edge_file_csc_cell(edgeList, nodesize, k=5):
    row=[]
    col=[]
    data=[]

    
    for edge in edgeList:
        row.append(edge.row)
        col.append(edge.col)
        data.append(1.0)
        row.append(edge.col)
        col.append(edge.row)
        data.append(1.0)

    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(nodesize, nodesize))
    
    #python output
    # return mtx, tfDict

    #Output for matlab
    return mtx, row, col, data



# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, geneList, geneDict, discreteFlag=False):
    samplelist=[]
    featurelist=[]
    data =[]
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
                    if discreteFlag:
                        if tmplist[item]>=avgtmp:
                            data.append(1)
                        else:
                            data.append(0)
                    else:
                        data.append(float(tmplist[item]))
                    data_count += 1
            count += 1
    f.close()
    # As dream: rows as genes, columns as samples: This is transpose of the original scRNA data
    feature = scipy.sparse.csr_matrix((data, (featurelist, samplelist)), shape=(len(selectList),count))  

    # For Matlab
    dim2out = [[0.0] * len(selectList) for i in range(count)]
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

    return feature, dim2out, dim2outD


# For node as cell
# Load gene expression into sparse matrix
def read_feature_file_sparse_cell(filename, geneList, geneDict, discreteFlag=False):
    samplelist=[]
    featurelist=[]
    data =[]
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
                    if discreteFlag:
                        if tmplist[item]>=avgtmp:
                            data.append(1)
                        else:
                            data.append(0)
                    else:
                        data.append(float(tmplist[item]))
                    data_count += 1
            count += 1
    f.close()
    # As dream: rows as cells, columns as genes: This is transpose of the original scRNA data
    feature = scipy.sparse.csr_matrix((data, (samplelist, featurelist)), shape=(count,len(selectList)))  

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

    return feature, dim2out, dim2outD


def read_edge_file_dict(filename, geneDict):
    graphdict={}
    tdict={}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]
            if end1 in geneDict and end2 in geneDict:
                tdict[geneDict[end1]]=""
                tdict[geneDict[end2]]=""
                if geneDict[end1] in graphdict:
                    tmplist = graphdict[geneDict[end1]]
                else:
                    tmplist = []
                tmplist.append(geneDict[end2])
                graphdict[geneDict[end1]]= tmplist
            count += 1
    f.close()
    #check and get full matrix
    for i in range(len(geneDict)):
        if i not in tdict:
            graphdict[i]=[]
    return graphdict

def read_edge_file_dict_cell(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge.row
        end2 = edge.col
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict

networkname=args.network_name
expressionname=args.expression_name
if args.network_name=='ttrust':
    networkname = 'trrust_rawdata.human.tsv'

if args.expression_name=='TGFb':
    expressionname = 'HMLE_TGFb_day_8_10.csv'
    # expressionname = 'HMLE_TGFb_day_8_10_part.csv'
elif args.expression_name=='sci-CAR':
    expressionname = 'sci-CAR.csv'
elif args.expression_name=='sci-CAR_D':
    expressionname = 'sci-CAR_D.csv'
elif args.expression_name=='test':
    expressionname = 'test_data.csv'

out_folder = "data/sc/"+args.expression_name+"/"
edge_filename    = "/home/wangjue/biodata/scData/network/"+networkname
feature_filename = "/home/wangjue/biodata/scData/"+expressionname
# edge_filename    = "data/"+networkname
# feature_filename = "data/"+expressionname

geneList, geneDict = preprocess_network(edge_filename, feature_filename)

#only python
# graphcsc, tfDict = read_edge_file_csc(edge_filename, geneDict)
# feature = read_feature_file_sparse(feature_filename, geneList, geneDict)

#python and matlab
if args.graph_type=='gene':
    graphcsc, tfDict, rowO, colO, dataO  = read_edge_file_csc(edge_filename, geneDict)
    feature, dim2out, dim2outD = read_feature_file_sparse(feature_filename, geneList, geneDict, discreteFlag=args.discrete_flag)
    graphdict = read_edge_file_dict(edge_filename, geneDict)
    outname = args.expression_name
elif args.graph_type=='cell':
    #First generate feature
    feature, dim2out, dim2outD = read_feature_file_sparse_cell(feature_filename, geneList, geneDict, discreteFlag=args.discrete_flag)
    edgeList = cal_distanceMatrix(feature, k=5)
    graphcsc, rowO, colO, dataO  = read_edge_file_csc_cell(edgeList, feature.shape[0], k=5)
    graphdict = read_edge_file_dict_cell(edgeList, feature.shape[0] )
    outname = args.expression_name + '.cell'

x = feature[0:100]
tx = feature[0:100]
allx = feature[100:]
# allx = feature
testindex = ""
for i in range(100):
    testindex = testindex + str(i) + "\n"

if args.graph_type=='gene':
    with open(out_folder+args.expression_name+"gene.txt",'w') as fw:
        count = 0
        for item in geneList:
            fw.write(str(count)+"\t"+item+"\n")
            count += 1

    with open(out_folder+args.expression_name+"TF.txt",'w') as fw:
        count = 0
        for key in tfDict:
            fw.write(key+"\t"+str(tfDict[key])+"\n")
            count += 1

pickle.dump(allx, open( out_folder+"ind."+outname+".allx", "wb" ) )
pickle.dump(graphcsc, open( out_folder+"ind."+outname+".csc", "wb" ) )

pickle.dump(x, open( out_folder+"ind."+outname+".x", "wb" ) )
pickle.dump(tx, open( out_folder+"ind."+outname+".tx", "wb" ) )
pickle.dump(graphdict, open( out_folder+"ind."+outname+".graph", "wb" ) )
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

with open(out_folder+outname+'.row.csv','w') as fw:
    for item in rowO:
        fw.write(str(item)+"\n")
fw.close()

with open(out_folder+outname+'.col.csv','w') as fw:
    for item in colO:
        fw.write(str(item)+"\n")
fw.close()

with open(out_folder+outname+'.data.csv','w') as fw:
    for item in dataO:
        fw.write(str(item)+"\n")
fw.close()

