#Used for generate sci-CAR and other data to the data format we used
#Original: line as the gene, column as the cell, first column is the gene name, first line is the cell name
#Output:   line as the cell, column as the gene, first line is the gene name.
#
#Usage:
# inputfile, inputfileD
#
import numpy as np

# inputfile  = '/home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.txt'
# outputfile = '/home/wangjue/biodata/scData/sci-CAR.csv'
inputfile  = '/home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.dic'
outputfile = '/home/wangjue/biodata/scData/sci-CAR_D.csv'
outputfileCellName = '/home/wangjue/biodata/scData/sci-CAR.cellname.txt'
cellcount = 1414
# Original:
genecount = 25178
# Discretization:
# genecount = 19467 

cellNames = []
geneNamesLine = ''

#cell as the row, col as the gene
contentArray = [[0.0] * genecount for i in range(cellcount)]
contentArray = np.asarray(contentArray)

count = -1
with open(inputfile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        words = line.split()
        if count == -1:
            for word in words:
                cellNames.append(word)
        else:
            colcount = -1
            for word in words:
                if colcount == -1:
                    geneNamesLine = geneNamesLine + word + ','
                else:
                    contentArray[colcount,count] = word
                colcount+=1
        count += 1
    f.close()

with open(outputfile, 'w') as fw:
    fw.write(geneNamesLine+'\n')
    for i in range(contentArray.shape[0]):
        for j in range(contentArray.shape[1]):
            fw.write(str(contentArray[i][j])+',')
        fw.write('\n')
    fw.close()