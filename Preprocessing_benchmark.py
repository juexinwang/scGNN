# For benchmark preprocessing usage:
#
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/9.Chung/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/9.Chung.csv --cellcount 317 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/11.Kolodziejczyk/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/11.Kolodziejczyk.csv --cellcount 704 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/12.Klein/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/12.Klein.csv --cellcount 2717 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/13.Zeisel/T2000_UsingOriginalMatrix/T2000_expression.txt  --outputfile /home/wangjue/biodata/scData/13.Zeisel.csv --cellcount 3005 --genecount 2000 --split space --cellheadflag False

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, default='/home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.dic',
                    help='inputfile name')
parser.add_argument('--outputfile', type=str, default='/home/wangjue/biodata/scData/sci-CAR_LTMG.csv',
                    help='outputfile name')
parser.add_argument('--cellcount', type=int, default=317,
                    help='total cell count')
parser.add_argument('--genecount', type=int, default=2000,
                    help='total gene count')
parser.add_argument('--split', type=str, default='space',
                    help='comma/blank')
parser.add_argument('--cellheadflag', type=bool, default=False,
                    help='True/False')
args = parser.parse_args()

inputfile = args.inputfile
outputfile = args.outputfile
cellcount = args.cellcount
genecount = args.genecount
splitChar = ''
if args.split == 'space':
    splitChar = ''
elif args.split == 'comma':
    splitChar = ',' 

geneNamesLine = ''

#cell as the row, col as the gene
contentArray = [[0.0] * genecount for i in range(cellcount)]
contentArray = np.asarray(contentArray)

count = -1
with open(inputfile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if splitChar == '':
            words = line.split()
        else:
            words = line.split(splitChar)
        if count == -1:
            colcount = -1
            for word in words:
                colcount += 1
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
    fw.write(geneNamesLine[:-1]+'\n')
    for i in range(contentArray.shape[0]):
        tmpStr = ''
        for j in range(contentArray.shape[1]):
            tmpStr = tmpStr + str(contentArray[i][j])+','
        fw.write(tmpStr[:-1]+'\n')
    fw.close()
