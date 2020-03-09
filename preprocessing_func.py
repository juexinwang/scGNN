import numpy as np
import pandas as pd
import os.path
import scipy.sparse as sp
import scipy.io

def preprocessing(dir,datasetName,transform='log',cellRatio=0.99,geneRatio=0.95,geneCriteria='variance',geneSelectnum=2000):
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

    genelist = []
    celllist = []
    datalist = []
    # Preprocessing before generating the whole data strcture
    tmpgenelist = []
    tmpcelllist = []
    tmpdatalist = []
    oldcellindex = -1
    cellNum = 0
    for index, row in df.iterrows():
        if not (row[1]-1) == oldcellindex:
            if len(tmpgenelist) >= len(genes)*(1-cellRatio):
                genelist.extend(tmpgenelist)
                tmpcelllist = []
                for i in range(len(tmpgenelist)):
                    tmpcelllist.append(cellNum)
                celllist.extend(tmpcelllist)
                datalist.extend(tmpdatalist)
                cellNamelist.append(index)
                cellNum += 1
            tmpgenelist = []            
            tmpdatalist = []
            oldcellindex = row[1]-1

        tmpgenelist.append(row[0]-1)
        tmpdata = row[2]
        if transform == 'log':
            tmpdatalist.append(np.log(tmpdata+1))
        elif transform == None:
            tmpdatalist.append(tmpdata)
    
    #post processing
    if len(tmpgenelist) >= len(genes)*(1-cellRatio):
        genelist.extend(tmpgenelist)
        tmpcelllist = []
        for i in range(len(tmpgenelist)):
            tmpcelllist.append(cellNum)
        celllist.extend(tmpcelllist)
        datalist.extend(tmpdatalist)
        cellNamelist.append(index)
        cellNum += 1

    print('After preprocessing, {} cells remaining'.format(len(cellNamelist)))

    # Now work on genes:
    
    data = scipy.sparse.csr_matrix((datalist, (genelist, celllist)), shape=(len(genes),len(cells))).tolil()
    return data