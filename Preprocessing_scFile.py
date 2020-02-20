#Used for generate sci-CAR and other data to the data format we used
#Original: line as the gene, column as the cell, first column is the gene name, first line is the cell name
#Output:   line as the cell, column as the gene, first line is the gene name.
#
#Usage:
#python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.txt\
#--outputfile /home/wangjue/biodata/scData/sci-CAR.csv --outputfileCellName /home/wangjue/biodata/scData/sci-CAR.cellname.txt\
#--cellcount 1414 --genecount 25178 --split space --cellheadflag True
#
#python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.dic\
#--outputfile /home/wangjue/biodata/scData/sci-CAR_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/sci-CAR.cellname.txt\
#--cellcount 1414 --genecount 19467 --split space --cellheadflag True
#
# Benchmark: Old, not use it at all
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/1.Biase/Biase_expression.csv --outputfile /home/wangjue/biodata/scData/1.Biase.csv --outputfileCellName /home/wangjue/biodata/scData/1.Biase.cellname.txt --cellcount 49 --genecount 25737 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/2.Yan/Yan_expression.csv --outputfile /home/wangjue/biodata/scData/2.Yan.csv --outputfileCellName /home/wangjue/biodata/scData/2.Yan.cellname.txt --cellcount 90 --genecount 20214 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/3.Goolam/Goolam_expression.csv --outputfile /home/wangjue/biodata/scData/3.Goolam.csv --outputfileCellName /home/wangjue/biodata/scData/3.Goolam.cellname.txt --cellcount 124 --genecount 41480 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/4.Deng/Deng_expression.csv --outputfile /home/wangjue/biodata/scData/4.Deng.csv --outputfileCellName /home/wangjue/biodata/scData/4.Deng.cellname.txt --cellcount 268 --genecount 22457 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_expression.csv --outputfile /home/wangjue/biodata/scData/5.Pollen.csv --outputfileCellName /home/wangjue/biodata/scData/5.Pollen.cellname.txt --cellcount 301 --genecount 23730 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/6.Kolodziejczyk/Kolodziejczyk_expression.csv --outputfile /home/wangjue/biodata/scData/6.Kolodziejczyk.csv --outputfileCellName /home/wangjue/biodata/scData/6.Kolodziejczyk.cellname.txt --cellcount 704 --genecount 38653 --split comma --cellheadflag False
#
# Updated Benchmark:
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/1.Biase/Biase_expression.csv --outputfile /home/wangjue/biodata/scData/1.Biase.csv --outputfileCellName /home/wangjue/biodata/scData/1.Biase.cellname.txt --cellcount 49 --genecount 25737 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/2.Li/Li_expression.txt --outputfile /home/wangjue/biodata/scData/2.Li.csv --outputfileCellName /home/wangjue/biodata/scData/2.Li.cellname.txt --cellcount 60 --genecount 180253 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/3.Treutlein/Treutlein_expression.csv --outputfile /home/wangjue/biodata/scData/3.Treutlein.csv --outputfileCellName /home/wangjue/biodata/scData/3.Treutlein.cellname.txt --cellcount 80 --genecount 23271 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/4.Yan/Yan_expression.csv --outputfile /home/wangjue/biodata/scData/4.Yan.csv --outputfileCellName /home/wangjue/biodata/scData/4.Yan.cellname.txt --cellcount 90 --genecount 20214 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/5.Goolam/Goolam_expression.csv --outputfile /home/wangjue/biodata/scData/5.Goolam.csv --outputfileCellName /home/wangjue/biodata/scData/5.Goolam.cellname.txt --cellcount 124 --genecount 41480 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/6.Guo/Guo_expression.txt --outputfile /home/wangjue/biodata/scData/6.Guo.csv --outputfileCellName /home/wangjue/biodata/scData/6.Guo.cellname.txt --cellcount 148 --genecount 36188 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/7.Deng/Deng_expression.csv --outputfile /home/wangjue/biodata/scData/7.Deng.csv --outputfileCellName /home/wangjue/biodata/scData/7.Deng.cellname.txt --cellcount 268 --genecount 22457 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/8.Pollen/Pollen_expression.csv --outputfile /home/wangjue/biodata/scData/8.Pollen.csv --outputfileCellName /home/wangjue/biodata/scData/8.Pollen.cellname.txt --cellcount 301 --genecount 23730 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/9.Chung/Chung_expression_filtered.txt --outputfile /home/wangjue/biodata/scData/9.Chung.csv --outputfileCellName /home/wangjue/biodata/scData/9.Chung.cellname.txt --cellcount 317 --genecount 57915 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/10.Usoskin/Usoskin_expression.csv --outputfile /home/wangjue/biodata/scData/10.Usoskin.csv --outputfileCellName /home/wangjue/biodata/scData/10.Usoskin.cellname.txt --cellcount 622 --genecount 25334 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/11.Kolodziejczyk/Kolodziejczyk_expression.csv --outputfile /home/wangjue/biodata/scData/11.Kolodziejczyk.csv --outputfileCellName /home/wangjue/biodata/scData/11.Kolodziejczyk.cellname.txt --cellcount 704 --genecount 38653 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/12.Klein/Klein_expression.csv --outputfile /home/wangjue/biodata/scData/12.Klein.csv --outputfileCellName /home/wangjue/biodata/scData/12.Klein.cellname.txt --cellcount 2717 --genecount 24175 --split comma --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/13.Zeisel/Zeisel_expression.csv --outputfile /home/wangjue/biodata/scData/13.Zeisel.csv --outputfileCellName /home/wangjue/biodata/scData/13.Zeisel.cellname.txt --cellcount 3005 --genecount 19972 --split comma --cellheadflag False
#
#
# MMP LTMG
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/MMP/MMPbasal_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/MMPbasal_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/MMPbasal_LTMG.cellname.txt --cellcount 4394 --genecount 3784 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/MMP/MMPbasal_all_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/MMPbasal_all_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/MMPbasal_all_LTMG.cellname.txt --cellcount 5432 --genecount 21299 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/MMP/T2000_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/MMPbasal_2000_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/MMPbasal_2000_LTMG.cellname.txt --cellcount 5432 --genecount 2000 --split space --cellheadflag False
#
# 4,5,7,8
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/4.Yan/T2000_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/4.Yan_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/4.Yan_LTMG.cellname.txt --cellcount 90 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/5.Goolam/T2000_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/5.Goolam_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/5.Goolam_LTMG.cellname.txt --cellcount 124 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/7.Deng/T2000_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/7.Deng_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/7.Deng_LTMG.cellname.txt --cellcount 268 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/8.Pollen/T2000_OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/8.Pollen_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/8.Pollen_LTMG.cellname.txt --cellcount 301 --genecount 2000 --split space --cellheadflag False
#
# 12,13,20,30
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/12.Klein/OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/12.Klein_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/12.Klein_LTMG.cellname.txt --cellcount 2717 --genecount 21221 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/13.Zeisel/OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/13.Zeisel_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/13.Zeisel_LTMG.cellname.txt --cellcount 3005 --genecount 21221 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/20.10X_2700_seurat/OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/20.10X_2700_seurat_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/20.10X_2700_seurat_LTMG.cellname.txt --cellcount 2698 --genecount 21221 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/AnjunBenchmark/30.Schafer/OneSign_LTMG.txt --outputfile /home/wangjue/biodata/scData/30.Schafer_LTMG.csv --outputfileCellName /home/wangjue/biodata/scData/30.Schafer_LTMG.cellname.txt --cellcount 2552 --genecount 21221 --split space --cellheadflag False
#
# Gradient
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T1000/T1000Discretization_LTMG.txt --outputfile /home/wangjue/biodata/scData/T1000_LTMG.csv --cellcount 704 --genecount 1000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T2000/T2000Discretization_LTMG.txt --outputfile /home/wangjue/biodata/scData/T2000_LTMG.csv --cellcount 704 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T4000/T4000Discretization_LTMG.txt --outputfile /home/wangjue/biodata/scData/T4000_LTMG.csv --cellcount 704 --genecount 4000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T8000/T8000Discretization_LTMG.txt --outputfile /home/wangjue/biodata/scData/T8000_LTMG.csv --cellcount 704 --genecount 8000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T1000/T1000_expression_matrix.txt --outputfile /home/wangjue/biodata/scData/T1000.csv --cellcount 704 --genecount 1000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T2000/T2000_expression_matrix.txt --outputfile /home/wangjue/biodata/scData/T2000.csv --cellcount 704 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T4000/T4000_expression_matrix.txt --outputfile /home/wangjue/biodata/scData/T4000.csv --cellcount 704 --genecount 4000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/gradient/11.Kolodziejczyk/T8000/T8000_expression_matrix.txt --outputfile /home/wangjue/biodata/scData/T8000.csv --cellcount 704 --genecount 8000 --split space --cellheadflag False

#Use 2000 data
#TODO: will change the sparse saving and loading later
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/1.Biase/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/1.Biase.csv --cellcount 49 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/2.Li/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/2.Li.csv --cellcount 60 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/3.Treutlein/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/3.Treutlein.csv --cellcount 80 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/4.Yan/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/4.Yan.csv --cellcount 90 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/5.Goolam/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/5.Goolam.csv --cellcount 124 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/6.Guo/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/6.Guo.csv --cellcount 148 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/7.Deng/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/7.Deng.csv --cellcount 268 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/8.Pollen/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/8.Pollen.csv --cellcount 301 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/9.Chung/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/9.Chung.csv --cellcount 317 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/10.Usoskin/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/10.Usoskin.csv --cellcount 622 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/11.Kolodziejczyk/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/11.Kolodziejczyk.csv --cellcount 704 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/12.Klein/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/12.Klein.csv --cellcount 2717 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/13.Zeisel/T2000_UsingOriginalMatrix/T2000_expression.txt  --outputfile /home/wangjue/biodata/scData/13.Zeisel.csv --cellcount 3005 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/20.10X_2700_seurat/T2000_UsingOriginalMatrix/T2000_expression.txt  --outputfile /home/wangjue/biodata/scData/20.10X_2700_seurat.csv --cellcount 2700 --genecount 2000 --split space --cellheadflag False
# python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/30.Schafer/T2000_UsingOriginalMatrix/T2000_expression.txt  --outputfile /home/wangjue/biodata/scData/30.Schafer.csv --cellcount 2552 --genecount 2000 --split space --cellheadflag False

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, default='/home/wangjue/biodata/scData/scRNA_And_scATAC_Files/Processed_Data/GeneSymbolMat.dic',
                    help='inputfile name')
parser.add_argument('--outputfile', type=str, default='/home/wangjue/biodata/scData/sci-CAR_LTMG.csv',
                    help='outputfile name')
parser.add_argument('--cellcount', type=int, default=1414,
                    help='total cell count')
parser.add_argument('--genecount', type=int, default=19467,
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
    fw.write(geneNamesLine+'\n')
    for i in range(contentArray.shape[0]):
        for j in range(contentArray.shape[1]):
            fw.write(str(contentArray[i][j])+',')
        fw.write('\n')
    fw.close()
