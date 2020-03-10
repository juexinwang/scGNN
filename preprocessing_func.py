import numpy as np
import pandas as pd
import os.path
import scipy.sparse as sp
import scipy.io

def preprocessing(dir,datasetName,csvFilename,transform='log',cellRatio=0.99,geneRatio=0.95,geneCriteria='variance',geneSelectnum=2000):
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
            if len(tmpgenelist) >= len(genes)*(1-cellRatio):
                for i in range(len(tmpgenelist)):
                    tmplist = expressionDict[tmpgenelist[0]]
                    tmplist.append(tmpdatalist[i])
                    expressionDict[tmpgenelist[0]] = tmplist

                    tmplist = expressionCellDict[tmpgenelist[0]]
                    tmplist.append(cellNum)
                    expressionCellDict[tmpgenelist[0]] = tmplist

                cellNamelist.append(cellNum)
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
            tmplist = expressionDict[tmpgenelist[0]]
            tmplist.append(tmpdatalist[i])
            expressionDict[tmpgenelist[0]] = tmplist

            tmplist = expressionCellDict[tmpgenelist[0]]
            tmplist.append(cellNum)
            expressionCellDict[tmpgenelist[0]] = tmplist

        cellNamelist.append(cellNum)
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

    tmpChooseIndex = np.argsort(finalList)[:geneSelectnum]
    tmpChooseIndex = tmpChooseIndex.tolist()

    genelist = []
    celllist = []
    datalist = []

    # output
    outList = []
    header = 'Gene_ID'
    for i in range(len(cellNamelist)):
        header = header + ',' + cells[0][cellNamelist[i]]
    outList.append(header)

    for index in tmpChooseIndex:
        geneindex = geneNamelist[index]        
        clist = expressionCellDict[geneindex]
        elist = expressionDict[geneindex]
        for i in range(len(elist)):
            genelist.append(geneindex)
            celllist.append(clist[i])
            datalist.append(elist[i]) 
        tmpline = ''
        for i in range(len(cellNamelist)):
            odata = 0.0
            #TODO can be improved
            for j in range(len(clist)):
                if cellNamelist[i] == clist[j]:
                    odata = elist[j]
                    break
                elif cellNamelist[i] < clist[j]:
                    break
            tmpline = tmpline + ',' + str(odata)
        outList.append(tmpline)

    with open(csvFilename,'w') as fw:
        fw.writelines(outList)
        fw.close()

    data = scipy.sparse.csr_matrix((datalist, (genelist, celllist)), shape=(geneSelectnum,len(cellNamelist))).tolil()
    return data

def generatingCSV():
    return