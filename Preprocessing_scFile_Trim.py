#Used for generate MMPfiles
#
#Usage:
# normalized results:
# python Preprocessing_scFile_Trim.py --inputfile /home/wangjue/biodata/scData/MMP/GSM2388072_basal_bone_marrow.filtered_normalized_counts.csv --outputfile /home/wangjue/biodata/scData/MMPbasal.csv --outputfileCellName /home/wangjue/biodata/scData/MMPbasal.cellname.txt --split dot
# python Preprocessing_scFile_Trim.py --inputfile /home/wangjue/biodata/scData/MMP/GSM2388073_epo_bone_marrow.filtered_normalized_counts.csv --outputfile /home/wangjue/biodata/scData/MMPepo.csv --outputfileCellName /home/wangjue/biodata/scData/MMPepo.cellname.txt --split dot
#
#Raw counts:
# python Preprocessing_scFile_Trim.py --inputfile /home/wangjue/biodata/scData/MMP/bBM.raw_umifm_counts.csv --outputfile /home/wangjue/biodata/scData/MMPbasal.csv --outputfileCellName /home/wangjue/biodata/scData/MMPbasal.cellname.txt --split dot
# python Preprocessing_scFile_Trim.py --inputfile /home/wangjue/biodata/scData/MMP/eBM.raw_umifm_counts.csv --outputfile /home/wangjue/biodata/scData/MMPepo.csv   --outputfileCellName /home/wangjue/biodata/scData/MMPepo.cellname.txt   --split dot


import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, default='/home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.dic',
                    help='inputfile name')
parser.add_argument('--outputfile', type=str, default='/home/wangjue/biodata/scData/sci-CAR_LTMG.csv',
                    help='outputfile name')
parser.add_argument('--outputfileCellName', type=str, default='/home/wangjue/biodata/scData/sci-CAR.cellname.txt',
                    help='outputfile cell name')
parser.add_argument('--split', type=str, default='dot',
                    help='dot/blank')
args = parser.parse_args()

inputfile = args.inputfile
outputfile = args.outputfile
outputfileCellName = args.outputfileCellName
splitChar = ''
if args.split == 'space':
    splitChar = ''
elif args.split == 'dot':
    splitChar = ',' 

geneNameStr = ''
geneNameDict = {}
cellNames = []

geneNameIndexStart = 5
outList = []

#cell as the row, col as the gene

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
            colcount = 0
            for word in words:
                if colcount >= geneNameIndexStart:
                    # select genes as in varID
                    # https://www.nature.com/articles/s41592-019-0632-3
                    if not word.startswith('mt') and not word.startswith('Rpl') and not word.startswith('Rps') and not word.startswith('Gm'):
                        geneNameDict[colcount] = word
                        geneNameStr = geneNameStr + word + ','
                colcount += 1
        else:
            colcount = 0
            outStr = ''
            for word in words:
                if colcount == 0:
                    cellNames.append(word)
                else:
                    if colcount in geneNameDict:
                        outStr = outStr + word + ',' 
                colcount+=1
            outList.append(outStr)
        count += 1
    f.close()

with open(outputfile, 'w') as fw:
    fw.write(geneNameStr+'\n')
    for item in outList:
        fw.write(item+'\n')
    fw.close()

with open(outputfileCellName, 'w') as fw:
    for cell in cellNames:
        fw.write(cell+'\n')
    fw.close()