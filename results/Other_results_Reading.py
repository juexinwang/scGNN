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
    'MMPbasal_2000',
    'MMPbasal_2000 --discreteTag',
    'MMPbasal_2000_LTMG',
    '4.Yan',
    '4.Yan --discreteTag',
    '4.Yan_LTMG',
    '5.Goolam',
    '5.Goolam --discreteTag',
    '5.Goolam_LTMG',
    '7.Deng',
    '7.Deng --discreteTag',
    '7.Deng_LTMG',
    '8.Pollen',
    '8.Pollen --discreteTag',
    '8.Pollen_LTMG',
    '11_Kolodziejczyk',
    '11_Kolodziejczyk --discreteTag'
    ]
    #TODO: we wait for 11.Kolodziejczyk_LTMG

if args.imputeMode:
    pyStr = 'Other_results_impute.py'
    npyList = [
        '../otherResults/SAUCIE_I/'
        ]
else:
    pyStr = 'Other_results_celltype.py'
    npyList = [
        '../otherResults/SAUCIE/'
        ]

npyStr = npyList[args.methodName]

benchmarkStr = ''

if args.runMode:
    labelFileDir = '/home/wangjue/biodata/scData/AnjunBenchmark/'
    cellFileDir  = '/home/wangjue/biodata/scData/'
    cellIndexDir = '/home/wangjue/myprojects/scGNN/data/sc/'
else:
    labelFileDir = '/home/jwang/data/scData/'
    cellFileDir  = '/home/jwang/data/scData/'
    cellIndexDir = '/home/jwang/data/scData/'
    
if int(args.methodName/3)==1:
    benchmarkStr = ' --benchmark '\
        '--labelFilename ' + labelFileDir + '4.Yan/Yan_cell_label.csv '\
        '--cellFilename ' + cellFileDir + '4.Yan.cellname.txt '\
        '--cellIndexname ' + cellIndexDir + '4.Yan/ind.4.Yan.cellindex.txt '\
        '--n-clusters 7 '
elif int(args.methodName/3)==2:
    benchmarkStr = ' --benchmark '\
        '--labelFilename ' + labelFileDir + '5.Goolam/Goolam_cell_label.csv '\
        '--cellFilename ' + cellFileDir + '5.Goolam.cellname.txt '\
        '--cellIndexname ' + cellIndexDir + '5.Goolam/ind.5.Goolam.cellindex.txt '\
        '--n-clusters 5 '
elif int(args.methodName/3)==3:
    benchmarkStr = ' --benchmark '\
        '--labelFilename ' + labelFileDir + '7.Deng/Deng_cell_label.csv '\
        '--cellFilename ' + cellFileDir + '7.Deng.cellname.txt '\
        '--cellIndexname ' + cellIndexDir + '7.Deng/ind.7.Deng.cellindex.txt '\
        '--n-clusters 10 '
elif int(args.methodName/3)==4:
    benchmarkStr = ' --benchmark '\
        '--labelFilename ' + labelFileDir + '8.Pollen/Pollen_cell_label.csv '\
        '--cellFilename ' + cellFileDir + '8.Pollen.cellname.txt '\
        '--cellIndexname ' + cellIndexDir + '8.Pollen/ind.8.Pollen.cellindex.txt '\
        '--n-clusters 11 '
elif int(args.methodName/3)==5:
    benchmarkStr = ' --benchmark '\
        '--labelFilename ' + labelFileDir + '11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
        '--cellFilename ' + cellFileDir + '11.Kolodziejczyk.cellname.txt '\
        '--cellIndexname ' + cellIndexDir + '11.Kolodziejczyk/ind.11.Kolodziejczyk.cellindex.txt '\
        '--n-clusters 3 '


if not args.runMode:
    if args.imputeMode:
        imputeStr = 'I'
    else:
        imputeStr = 'C'
    templateStr = "#! /bin/bash\n"\
    "######################### Batch Headers #########################\n"\
    "#SBATCH -A xulab\n"\
    "#SBATCH -p BioCompute               # use the BioCompute partition\n"\
    "#SBATCH -J R" + imputeStr + str(args.methodName) +              " \n"\
    "#SBATCH -o results-%j.out           # give the job output a custom name\n"\
    "#SBATCH -t 2-00:00                  # two days time limit\n"\
    "#SBATCH -N 1                        # number of nodes\n"\
    "#SBATCH -n 8                        # number of cores (AKA tasks)\n"\
    "#SBATCH --mem=64G\n"\
    "#################################################################\n"\
    "module load miniconda3\n"\
    "source activate conda_R\n"
    print(templateStr)

for datasetStr in datasetList:
    commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + benchmarkStr + ' --npyDir ' + npyStr
    if args.runMode:
        os.system(commandStr)
    else:
        print(commandStr)


