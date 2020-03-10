import numpy as np
import pandas as pd
import os.path
import scipy.sparse as sp
import scipy.io

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
        if len(tmplist) >= len(cells)*(1-geneRatio):
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

def loadscCSV(csvFilename):
    matrix = pd.read_csv(csvFilename,header=None, index_col=None)
    matrix = matrix.to_numpy()
    matrix = matrix[1:,1:]
    matrix = matrix.astype(float)
    return matrix