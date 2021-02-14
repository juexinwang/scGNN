import time
import argparse
import numpy as np
import pandas as pd
import pickle
import os.path
import scipy.sparse as sp
import scipy.io

parser = argparse.ArgumentParser(description='Main Entrance of scGNN')
parser.add_argument('--datasetName', type=str, default='481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/MMPbasal/MMPbasal_all/MMPbasal_allgene/MMPbasal_allcell/MMPepo/MMPbasal_LTMG/MMPbasal_all_LTMG/MMPbasal_2000')
parser.add_argument('--datasetDir', type=str, default='/storage/htc/joshilab/wangjue/10x/6/',
                    help='Directory of data, default(/home/wangjue/biodata/scData/10x/6/)')
parser.add_argument('--nonfilterCSVTag', action='store_true', default=False,
                    help='Not filter and generating CSV')
parser.add_argument('--inferLTMGTag', action='store_true', default=False,
                    help='Infer LTMG (Optional)')
parser.add_argument('--nonsparseOutTag', action='store_true', default=False,
                    help='Not use sparse coding')
parser.add_argument('--LTMGDir', type=str, default='/home/wangjue/biodata/scData/10x/6/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--expressionFile', type=str, default='Use_expression.csv',
                    help='expression File in csv')
parser.add_argument('--ltmgFile', type=str, default='ltmg.csv',
                    help='expression File in csv')
parser.add_argument('--filetype', type=str, default='10X',
                    help='select input filetype, 10X or CSV: default(10X)')
parser.add_argument('--delim', type=str, default='comma',
                    help='File delim type, comma or space: default(comma)')
# param
parser.add_argument('--transform', type=str, default='log',
                    help='Whether transform')
parser.add_argument('--cellRatio', type=float, default=0.99,
                    help='cell ratio')
parser.add_argument('--geneRatio', type=float, default=0.99,
                    help='gene ratio')
parser.add_argument('--geneCriteria', type=str, default='variance',
                    help='gene Criteria')
parser.add_argument('--geneSelectnum', type=int, default=2000,
                    help='select top gene numbers')
parser.add_argument('--transpose', action='store_true', default=False,
                    help='whether transpose or not')
parser.add_argument('--tabuCol', type=str, default='',
                    help='Not use some columns and setting their names split by ,')

args = parser.parse_args()
args.sparseOutTag = not args.nonsparseOutTag
args.filterCSVTag = not args.nonfilterCSVTag
# args.inferLTMGTag = not args.noninferLTMGTag
# print(args)


def preprocessing10X(dir, datasetName, csvFilename, transform='log', cellRatio=0.99, geneRatio=0.99, geneCriteria='variance', geneSelectnum=2000, sparseOut=True):
    '''
    preprocessing 10X data
    transform='log' or None
    '''
    filefolder = dir + datasetName + '/'
    if not os.path.exists(filefolder):
        print('Dataset ' + filefolder + ' not exists!')

    # Three files of 10x
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

    print('Input scRNA data in 10X is validated, start reading...')

    genes = pd.read_csv(featuresFilename, header=None, delim_whitespace=True)
    cells = pd.read_csv(barcodesFilename, header=None, delim_whitespace=True)
    df = pd.read_csv(expressionFilename, header=None,
                     skiprows=2, delim_whitespace=True)

    print('Data loaded, start filtering...')

    geneNamelist = []
    cellNamelist = []

    geneNameDict = {}
    cellNameDict = {}

    expressionDict = {}
    expressionCellDict = {}
    for i in range(len(genes)):
        expressionDict[i] = []
        expressionCellDict[i] = []

    # Preprocessing before generating the whole data strcture
    tmpgenelist = []
    tmpdatalist = []
    oldcellindex = -1
    cellNum = 0

    for row in df.itertuples():
        if row.Index % 1000000 == 0:
            print(str(row.Index)+' items in expression has been proceed.')
        if not (row[2]-1) == oldcellindex:
            if (row[2]-1) < oldcellindex:
                print('Potential error in 10X data: '+str(oldcellindex)+'!')
            if len(tmpgenelist) >= len(genes)*(1-cellRatio) and not oldcellindex == -1:
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

    # post processing
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
    finalList = []
    for i in range(len(genes)):
        tmplist = expressionDict[i]
        if len(tmplist) >= len(cellNamelist)*(1-geneRatio):
            geneNamelist.append(i)
            if geneCriteria == 'variance':
                finalList.append(-np.var(tmplist))

    print('After preprocessing, {} genes have {} nonzero'.format(
        len(geneNamelist), geneRatio))

    tmpChooseIndex = np.argsort(finalList)[:geneSelectnum]
    tmpChooseIndex = tmpChooseIndex.tolist()

    for i in range(len(tmpChooseIndex)):
        geneNameDict[geneNamelist[tmpChooseIndex[i]]] = i

    genelist = []
    celllist = []
    datalist = []

    outgenelist = []
    outcelllist = []

    # output
    outList = []
    header = 'Gene_ID'
    for i in range(len(cellNamelist)):
        # print('{}\t{}\t{}'.format(cellNamelist[i],cells[cellNamelist[i]],cells[cellNamelist[i]][0]))
        header = header + ',' + cells[0][cellNamelist[i]]
        outcelllist.append(cells[0][cellNamelist[i]])
    outList.append(header+'\n')

    for index in tmpChooseIndex:
        # print(index)
        geneindex = geneNamelist[index]
        clist = expressionCellDict[geneindex]
        elist = expressionDict[geneindex]

        # For output sparse purpose
        if sparseOut:
            for i in range(len(elist)):
                # print('{}*{}'.format(geneindex,geneNameDict[geneindex]))
                genelist.append(geneNameDict[geneindex])
                celllist.append(cellNameDict[clist[i]])
                datalist.append(elist[i])

        # print('*')
        tmpline = genes[0][index]
        outgenelist.append(tmpline)
        # print(str(len(cellNamelist))+' '+str(len(clist)))
        k = 0
        for l in range(len(cellNamelist)):
            for j in range(k, len(clist)):
                # print(j)
                if cellNamelist[l] == clist[j]:
                    tmpline = tmpline + ','
                    tmpline = tmpline + str(elist[j])
                    k = j+1
                    break
                elif cellNamelist[l] < clist[j]:
                    tmpline = tmpline + ','
                    tmpline = tmpline + str(0.0)
                    k = j
                    break

        size = tmpline.split(',')
        for i in range(len(size), len(cellNamelist)+1):
            tmpline = tmpline + ','
            tmpline = tmpline + str(0.0)

        outList.append(tmpline+'\n')
        size = tmpline.split(',')
        # For debug usage
        # print(str(index)+'*'+str(len(size)))

    with open(csvFilename, 'w') as fw:
        fw.writelines(outList)
        fw.close()
    print('Write CSV done')

    # For output sparse purpose
    if sparseOut:
        data = scipy.sparse.csr_matrix((datalist, (genelist, celllist)), shape=(
            len(tmpChooseIndex), len(cellNamelist))).tolil()
        pickle.dump(data, open(csvFilename.replace(
            '.csv', '_sparse.npy'), "wb"))
        print('Write sparse output done')

        with open(csvFilename.replace('.csv', '_gene.txt'), 'w') as f:
            f.writelines("%s\n" % gene for gene in outgenelist)
            f.close()

        with open(csvFilename.replace('.csv', '_cell.txt'), 'w') as f:
            f.writelines("%s\n" % cell for cell in outcelllist)
            f.close()


def preprocessingCSV(dir, datasetName, csvFilename, delim='comma', transform='log', cellRatio=0.99, geneRatio=0.99, geneCriteria='variance', geneSelectnum=2000, transpose=False, tabuCol=''):
    '''
    preprocessing CSV files:
    transform='log' or None
    '''
    expressionFilename = dir + datasetName
    if not os.path.exists(expressionFilename):
        print('Dataset ' + expressionFilename + ' not exists!')

    print('Input scRNA data in CSV format is validated, start reading...')

    tabuColList = []
    tmplist = tabuCol.split(",")
    for item in tmplist:
        tabuColList.append(item)

    df = pd.DataFrame()
    if delim == 'space':
        if len(tabuColList) == 0:
            df = pd.read_csv(expressionFilename, index_col=0,
                             delim_whitespace=True)
        else:
            df = pd.read_csv(expressionFilename, index_col=0, delim_whitespace=True,
                             usecols=lambda column: column not in tabuColList)
    elif delim == 'comma':
        if len(tabuColList) == 0:
            df = pd.read_csv(expressionFilename, index_col=0)
        else:
            df = pd.read_csv(expressionFilename, index_col=0,
                             usecols=lambda column: column not in tabuColList)
    print('Data loaded, start filtering...')
    if transpose == True:
        df = df.T
    df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]
    print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
    criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
    df2 = df1[df1.columns[criteriaGene]]
    print('After preprocessing, {} cells have {} nonzero'.format(
        df2.shape[1], geneRatio))
    criteriaSelectGene = df2.var(axis=1).sort_values()[-geneSelectnum:]
    df3 = df2.loc[criteriaSelectGene.index]
    if transform == 'log':
        df3 = df3.transform(lambda x: np.log(x + 1))
    df3.to_csv(csvFilename)


if __name__ == "__main__":
    start_time = time.time()

    # preprocessing
    if args.filterCSVTag:
        print('Step1: Start filter and generating CSV')
        if args.filetype == '10X':
            expressionFilename = args.LTMGDir+args.datasetName+'/'+args.expressionFile
            # data = preprocessing10X(args.datasetDir, args.datasetName, args.LTMGDir+args.datasetName+'/'+args.expressionFile, args.transform, args.cellRatio, args.geneRatio, args.geneCriteria, args.geneSelectnum)
            preprocessing10X(args.datasetDir, args.datasetName, expressionFilename, args.transform,
                             args.cellRatio, args.geneRatio, args.geneCriteria, args.geneSelectnum, args.sparseOutTag)
        elif args.filetype == 'CSV':
            expressionFilename = args.LTMGDir+args.expressionFile
            preprocessingCSV(args.datasetDir, args.datasetName, expressionFilename, args.delim, args.transform,
                             args.cellRatio, args.geneRatio, args.geneCriteria, args.geneSelectnum, args.transpose, args.tabuCol)

    if args.inferLTMGTag:
        from LTMG_R import *
        print('Step2: Start infer LTMG from CSV')
        if args.filetype == '10X':
            ltmgdir = args.LTMGDir+args.datasetName+'/'
        elif args.filetype == 'CSV':
            ltmgdir = args.LTMGDir
        # run LTMG in R
        runLTMG(ltmgdir+args.expressionFile, ltmgdir)

    print("Preprocessing Done. Total Running Time: %s seconds" %
          (time.time() - start_time))
