import time
import argparse
import numpy as np
import pandas as pd
import os.path
import scipy.sparse as sp
import scipy.io
from LTMG_R import *

parser = argparse.ArgumentParser(description='Main Entrance of scGNN')
parser.add_argument('--datasetName', type=str, default='481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/MMPbasal/MMPbasal_all/MMPbasal_allgene/MMPbasal_allcell/MMPepo/MMPbasal_LTMG/MMPbasal_all_LTMG/MMPbasal_2000')
parser.add_argument('--datasetDir', type=str, default='/storage/htc/joshilab/wangjue/10x/6/',
                    help='Directory of data, default(/home/wangjue/biodata/scData/10x/6/)')
parser.add_argument('--inferLTMGTag', action='store_true', default=False,
                    help='Whether infer LTMG')                   
parser.add_argument('--LTMGDir', type=str, default='/home/wangjue/biodata/scData/10x/6/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--expressionFile', type=str, default='Use_expression.csv',
                    help='expression File in csv')
parser.add_argument('--ltmgFile', type=str, default='ltmg.csv',
                    help='expression File in csv')
#param                    
parser.add_argument('--transform', type=str, default='log',
                    help='Whether transform')
parser.add_argument('--cellRatio', type=float, default=0.99,
                    help='cell ratio')
parser.add_argument('--geneRatio', type=float, default=0.99,
                    help='gene ratio')
parser.add_argument('--geneCriteria', type=str, default='variance',
                    help='gene Criteria')
parser.add_argument('--cellRatio', type=int, default=2000,
                    help='cell ratio')
                    
args = parser.parse_args()

def preprocessing(dir,datasetName,csvFilename,transform='log',cellRatio=0.99,geneRatio=0.99,geneCriteria='variance',geneSelectnum=2000):
    '''
    preprocessing:
    transform='log' or None
    '''
    filefolder = dir + datasetName + '/'
    if not os.path.exists(filefolder):
        print('Dataset '+ filefolder + ' not exists!')
    
    #Three files of 10x 
    featuresFilename = filefolder + 'features.tsv'
    if os.path.exists(featuresFilename+'.gz'):
        featuresFilename = featuresFilename+'.gz'
    elif not os.path.exists(featuresFilename):
        print('features.tsv or features.tsv.gz not exists!')

    barcodesFilename = filefolder + 'barcodes.tsv'
    if os.path.exists(barcodesFilename+'.gz'):
        barcodesFilename = barcodesFilename+'.gz'
    elif not os.path.exists(barcodesFilename):
        print('barcodes.tsv or barcodes.tsv.gz not exists!')

    expressionFilename = filefolder + 'matrix.mtx'
    if os.path.exists(expressionFilename+'.gz'):
        expressionFilename = expressionFilename+'.gz'
    elif not os.path.exists(expressionFilename):
        print('matrix.mtx or matrix.mtx.gz not exists!')

    genes = pd.read_csv(featuresFilename, header=None, delim_whitespace=True)
    cells = pd.read_csv(barcodesFilename, header=None, delim_whitespace=True)
    df    = pd.read_csv(expressionFilename, header=None, skiprows=2, delim_whitespace=True)

    geneNamelist = []
    cellNamelist = []

    geneNameDict = {}
    cellNameDict = {}

    expressionDict = {}
    expressionCellDict = {}
    for i in range(len(genes)):
        expressionDict[i]=[]
        expressionCellDict[i]=[]

    # Preprocessing before generating the whole data strcture
    tmpgenelist = []
    tmpdatalist = []
    oldcellindex = -1
    cellNum = 0

    for row in df.itertuples():
        if row.Index % 1000000 == 0:
            print(row.Index)
        if not (row[2]-1) == oldcellindex:
            if (row[2]-1)<oldcellindex:
                print(str(oldcellindex)+'!')
            if len(tmpgenelist) >= len(genes)*(1-cellRatio) and not oldcellindex==-1:
                for i in range(len(tmpgenelist)):
                    tmplist = expressionDict[tmpgenelist[i]]
                    tmplist.append(tmpdatalist[i])
                    expressionDict[tmpgenelist[0]] = tmplist

                    tmplist = expressionCellDict[tmpgenelist[i]]
                    tmplist.append(oldcellindex)
                    expressionCellDict[tmpgenelist[0]] = tmplist

                cellNamelist.append(oldcellindex)
                cellNameDict[oldcellindex] = cellNum
                cellNum += 1
            tmpgenelist = []            
            tmpdatalist = []
            oldcellindex = row[2]-1

        tmpgenelist.append(row[1]-1)
        tmpdata = row[3]
        if transform == 'log':
            tmpdatalist.append(np.log(tmpdata+1))
        elif transform == None:
            tmpdatalist.append(tmpdata)
    
    #post processing
    if len(tmpgenelist) >= len(genes)*(1-cellRatio):
        for i in range(len(tmpgenelist)):
            tmplist = expressionDict[tmpgenelist[i]]
            tmplist.append(tmpdatalist[i])
            expressionDict[tmpgenelist[0]] = tmplist

            tmplist = expressionCellDict[tmpgenelist[i]]
            tmplist.append(oldcellindex)
            expressionCellDict[tmpgenelist[0]] = tmplist

        cellNamelist.append(oldcellindex)
        cellNameDict[oldcellindex] = cellNum
        cellNum += 1
    
    print('After preprocessing, {} cells remaining'.format(len(cellNamelist)))

    # Now work on genes:
    finalList=[]
    for i in range(len(genes)):
        tmplist = expressionDict[i]
        if len(tmplist) >= len(cellNamelist)*(1-geneRatio):
            geneNamelist.append(i)
            if geneCriteria=='variance':
                finalList.append(-np.var(tmplist))
    
    print('After preprocessing, {} genes have {} nonzero'.format(len(geneNamelist),geneRatio))

    tmpChooseIndex = np.argsort(finalList)[:geneSelectnum]
    tmpChooseIndex = tmpChooseIndex.tolist()

    for i in range(len(tmpChooseIndex)):
        geneNameDict[geneNamelist[tmpChooseIndex[i]]]=i

    genelist = []
    celllist = []
    datalist = []

    # output
    outList = []
    header = 'Gene_ID'
    for i in range(len(cellNamelist)):
        # print('{}\t{}\t{}'.format(cellNamelist[i],cells[cellNamelist[i]],cells[cellNamelist[i]][0]))
        header = header + ',' + cells[0][cellNamelist[i]]
    outList.append(header+'\n')

    for index in tmpChooseIndex:
        # if index==4350:
        #     print(index)
        geneindex = geneNamelist[index]        
        clist = expressionCellDict[geneindex]
        elist = expressionDict[geneindex]
        # print(str(len(clist))+' '+str(len(elist)))
        for i in range(len(elist)):
            # print('{}*{}'.format(geneindex,geneNameDict[geneindex]))
            genelist.append(geneNameDict[geneindex])
            celllist.append(cellNameDict[clist[i]])
            datalist.append(elist[i]) 
        
        # print('*')
        tmpline = genes[0][index]
        # print(str(len(cellNamelist))+' '+str(len(clist)))
        k=0
        for l in range(len(cellNamelist)):
            for j in range(k,len(clist)):
                # print(j)
                if cellNamelist[l] == clist[j]:
                    tmpline = tmpline + ',' + str(elist[j])
                    k=j+1
                    break
                elif cellNamelist[l] < clist[j]:
                    tmpline = tmpline + ',' + str(0.0)
                    k=j
                    break
            if cellNamelist[l]>clist[-1]:
                tmpline = tmpline + ',' + str(0.0)

        outList.append(tmpline+'\n')

    with open(csvFilename,'w') as fw:
        fw.writelines(outList)
        fw.close()
    print('Write CSV done')

    data = scipy.sparse.csr_matrix((datalist, (genelist, celllist)), shape=(len(tmpChooseIndex),len(cellNamelist))).tolil()
    return data

if __name__ == "__main__":
    start_time = time.time()
    
    #preprocessing
    data = preprocessing(args.datasetDir, args.datasetName, args.LTMGDir+args.datasetName+'/'+args.expressionFile, args.transform, args.cellRatio, args.geneRatio, args.geneCriteria, args.geneSelectnum)
        
    if args.inferLTMGTag:
        #run LTMG in R
        runLTMG(args.LTMGDir+args.datasetName+'/'+args.expressionFile, args.LTMGDir+args.datasetName+'/'+args.ltmgFile)

    print("---Total Running Time: %s seconds ---" % (time.time() - start_time))