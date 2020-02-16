import os
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods')
parser.add_argument('--methodName', type=int, default=0, 
                    help="method used: 0-? 0: SAUICE")
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
parser.add_argument('--runMode',action='store_true', default=False, help="Run or prepare cluster script")
args = parser.parse_args()

# Note:
# Change a little bit of results_Reading.py to read results got from other methods 
# Generate results in python other than in shell for better organization
# We are not use runpy.run_path('main_result.py') for it is hard to pass arguments
# We are not use subprocess.call("python main_result.py", shell=True) for it runs scripts parallel
# So we use os.system('') here
datasetList = [
    '1.Biase',
    '1.Biase --discreteTag',
    '2.Li',
    '2.Li --discreteTag',
    '3.Treutlein',
    '3.Treutlein --discreteTag',
    '4.Yan',
    '4.Yan --discreteTag',
    '5.Goolam',
    '5.Goolam --discreteTag',
    '6.Guo',
    '6.Guo --discreteTag',
    '7.Deng',
    '7.Deng --discreteTag',
    '8.Pollen',
    '8.Pollen --discreteTag',
    '9.Chung',
    '9.Chung --discreteTag',
    '10.Usoskin',
    '10.Usoskin --discreteTag',
    '11.Kolodziejczyk',
    '11.Kolodziejczyk --discreteTag',
    '12.Klein',
    '12.Klein --discreteTag',
    '13.Zeisel',
    '13.Zeisel --discreteTag',
    'MMPbasal_2000',
    'MMPbasal_2000 --discreteTag'
    ]
    #TODO: we wait for 11.Kolodziejczyk_LTMG

if args.imputeMode:
    pyStr = 'Other_results_impute.py'
    npyList = [
        '../otherResults/SAUCIE_I/ --clusterTag',
        '../otherResults/MAGIC/'
        ]
else:
    pyStr = 'Other_results_celltype.py'
    npyList = [
        '../otherResults/SAUCIE/'
        ]

npyStr = npyList[args.methodName]

if args.runMode:
    labelFileDir = '/home/wangjue/biodata/scData/AllBench/'
    cellFileDir  = '/home/wangjue/biodata/scData/'
    cellIndexDir = '/home/wangjue/myprojects/scGNN/data/sc/'
else:
    labelFileDir = '/home/jwang/data/scData/'
    cellFileDir  = '/home/jwang/data/scData/'
    cellIndexDir = '/home/jwang/data/scData/'

def getBenchmarkStr(count):
    benchmarkStr = ''    
    if count == 0:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '1.Biase/Biase_cell_label.csv '\
                    '--n-clusters 3 '
    elif count == 1:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '2.Li/Li_cell_label.csv '\
                    '--n-clusters 9 '
    elif count == 2:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '3.Treutlein/Treutlein_cell_label.csv '\
                    '--n-clusters 5 '
    elif count == 3:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '4.Yan/Yan_cell_label.csv '\
                    '--n-clusters 7 '
    elif count == 4:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '5.Goolam/Goolam_cell_label.csv '\
                    '--n-clusters 5 '
    elif count == 5:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '6.Guo/Guo_cell_label.csv '\
                    '--n-clusters 9 '
    elif count == 6:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '7.Deng/Deng_cell_label.csv '\
                    '--n-clusters 10 '
    elif count == 7:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '8.Pollen/Pollen_cell_label.csv '\
                    '--n-clusters 11 '
    elif count == 8:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '9.Chung/Chung_cell_label.csv '\
                    '--n-clusters 4 '
    elif count == 9:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '10.Usoskin/Usoskin_cell.csv '\
                    '--n-clusters 11 '
    elif count == 10:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
                    '--n-clusters 3 '
    elif count == 11:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '12.Klein/Klein_cell_label.csv '\
                    '--n-clusters 4 '
    elif count == 12:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '13.Zeisel/Zeisel_cell_label.csv '\
                    '--n-clusters 7 '
    return benchmarkStr

if not args.runMode:
    if args.imputeMode:
        imputeStr = 'I'
    else:
        imputeStr = 'C'
    templateStr = "#! /bin/bash\n"\
    "######################### Batch Headers #########################\n"\
    "#SBATCH -A xulab\n"\
    "#SBATCH -p BioCompute               # use the BioCompute partition\n"\
    "#SBATCH -J O" + imputeStr + str(args.methodName) +              " \n"\
    "#SBATCH -o results-%j.out           # give the job output a custom name\n"\
    "#SBATCH -t 2-00:00                  # two days time limit\n"\
    "#SBATCH -N 1                        # number of nodes\n"\
    "#SBATCH -n 8                        # number of cores (AKA tasks)\n"\
    "#SBATCH --mem=64G\n"\
    "#################################################################\n"\
    "module load miniconda3\n"\
    "source activate conda_R\n"
    print(templateStr)

count = 0
for datasetStr in datasetList:
    commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + getBenchmarkStr(count) + ' --npyDir ' + npyStr
    if args.runMode:
        os.system(commandStr)
    else:
        print(commandStr)
    count += 1


