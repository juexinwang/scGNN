import os
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods')
parser.add_argument('--methodName', type=int, default=0, 
                    help="method used: 0-27")
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
parser.add_argument('--runMode',action='store_true', default=False, help="Run or prepare cluster script")
parser.add_argument('--splitMode', default=False, action='store_true',
                    help='whether split, used for long queue')
parser.add_argument('--batchStr', type=int, default=0, 
                    help="method used: 0-1")
args = parser.parse_args()

# Note:
# Generate results in python other than in shell for better organization
# We are not use runpy.run_path('main_result.py') for it is hard to pass arguments
# We are not use subprocess.call("python main_result.py", shell=True) for it runs scripts parallel
# So we use os.system('') here

if args.splitMode:
    #The split of batch, more batches, more parallel
    # if args.batchStr == 0:
    #     datasetList = [
    #     'MMPbasal_2000',
    #     'MMPbasal_2000 --discreteTag',
    #     'MMPbasal_2000_LTMG',
    #     '4.Yan',
    #     '4.Yan --discreteTag',
    #     '4.Yan_LTMG',
    #     '5.Goolam',
    #     '5.Goolam --discreteTag',
    #     '5.Goolam_LTMG',
    #     ]
    # elif args.batchStr == 1:
    #     datasetList = [
    #     '7.Deng',
    #     '7.Deng --discreteTag',
    #     '7.Deng_LTMG',
    #     '8.Pollen',
    #     '8.Pollen --discreteTag',
    #     '8.Pollen_LTMG',
    #     '11.Kolodziejczyk',
    #     '11.Kolodziejczyk --discreteTag'
    #     ]
    if args.batchStr == 0:
        datasetList = [
        'T1000',
        'T1000 --discreteTag',
        'T1000_LTMG'
        ]
    elif args.batchStr == 1:
        datasetList = [
        'T2000',
        'T2000 --discreteTag',
        'T2000_LTMG'
        ]
    elif args.batchStr == 2:
        datasetList = [
        'T4000',
        'T4000 --discreteTag',
        'T4000_LTMG'
        ]
    elif args.batchStr == 3:
        datasetList = [
        'T8000',
        'T8000 --discreteTag',
        'T8000_LTMG'
        ]
else:
    datasetList = [
    # 'MMPbasal_2000',
    # 'MMPbasal_2000 --discreteTag',
    # 'MMPbasal_2000_LTMG',
    # '4.Yan',
    # '4.Yan --discreteTag',
    # '4.Yan_LTMG',
    # '5.Goolam',
    # '5.Goolam --discreteTag',
    # '5.Goolam_LTMG',
    # '7.Deng',
    # '7.Deng --discreteTag',
    # '7.Deng_LTMG',
    # '8.Pollen',
    # '8.Pollen --discreteTag',
    # '8.Pollen_LTMG',
    # '11.Kolodziejczyk',
    # '11.Kolodziejczyk --discreteTag'
    'T1000',
    'T1000 --discreteTag',
    'T1000_LTMG',
    'T2000',
    'T2000 --discreteTag',
    'T2000_LTMG',
    'T4000',
    'T4000 --discreteTag',
    'T4000_LTMG',
    'T8000',
    'T8000 --discreteTag',
    'T8000_LTMG'
    ]


if args.imputeMode:
    pyStr = 'results_impute.py'
    npyList = [
        '../npyImputeG1B/',
        '../npyImputeG1E/',
        '../npyImputeG1F/',
        '../npyImputeN1B/',
        '../npyImputeN1E/',
        '../npyImputeN1F/',
        '../npyImputeG2B_AffinityPropagation/',
        '../npyImputeG2B_AgglomerativeClustering/',
        '../npyImputeG2B_Birch/',
        '../npyImputeG2B_KMeans/',
        '../npyImputeG2B_SpectralClustering/',
        '../npyImputeG2B/',
        '../npyImputeG2E_AffinityPropagation/',
        '../npyImputeG2E_AgglomerativeClustering/',
        '../npyImputeG2E_Birch/',
        '../npyImputeG2E_KMeans/',
        '../npyImputeG2E_SpectralClustering/',
        '../npyImputeG2E/',
        '../npyImputeG2F_AffinityPropagation/',
        '../npyImputeG2F_AgglomerativeClustering/',
        '../npyImputeG2F_Birch/',
        '../npyImputeG2F_KMeans/',
        '../npyImputeG2F_SpectralClustering/',
        '../npyImputeG2F/',
        '../npyImputeN2B_AffinityPropagation/',
        '../npyImputeN2B_AgglomerativeClustering/',
        '../npyImputeN2B_Birch/',
        '../npyImputeN2B_KMeans/',
        '../npyImputeN2B_SpectralClustering/',
        '../npyImputeN2B/',
        '../npyImputeN2E_AffinityPropagation/',
        '../npyImputeN2E_AgglomerativeClustering/',
        '../npyImputeN2E_Birch/',
        '../npyImputeN2E_KMeans/',
        '../npyImputeN2E_SpectralClustering/',
        '../npyImputeN2E/',
        '../npyImputeN2F_AffinityPropagation/',
        '../npyImputeN2F_AgglomerativeClustering/',
        '../npyImputeN2F_Birch/',
        '../npyImputeN2F_KMeans/',
        '../npyImputeN2F_SpectralClustering/',
        '../npyImputeN2F/'
        ]
else:
    pyStr = 'results_celltype.py'
    npyList = [
        '../npyG1B/',
        '../npyG1E/',
        '../npyG1F/',
        '../npyN1B/',
        '../npyN1E/',
        '../npyN1F/',
        '../npyG2B_AffinityPropagation/',
        '../npyG2B_AgglomerativeClustering/',
        '../npyG2B_Birch/',
        '../npyG2B_KMeans/',
        '../npyG2B_SpectralClustering/',
        '../npyG2B/',
        '../npyG2E_AffinityPropagation/',
        '../npyG2E_AgglomerativeClustering/',
        '../npyG2E_Birch/',
        '../npyG2E_KMeans/',
        '../npyG2E_SpectralClustering/',
        '../npyG2E/',
        '../npyG2F_AffinityPropagation/',
        '../npyG2F_AgglomerativeClustering/',
        '../npyG2F_Birch/',
        '../npyG2F_KMeans/',
        '../npyG2F_SpectralClustering/',
        '../npyG2F/',
        '../npyN2B_AffinityPropagation/',
        '../npyN2B_AgglomerativeClustering/',
        '../npyN2B_Birch/',
        '../npyN2B_KMeans/',
        '../npyN2B_SpectralClustering/',
        '../npyN2B/',
        '../npyN2E_AffinityPropagation/',
        '../npyN2E_AgglomerativeClustering/',
        '../npyN2E_Birch/',
        '../npyN2E_KMeans/',
        '../npyN2E_SpectralClustering/',
        '../npyN2E/',
        '../npyN2F_AffinityPropagation/',
        '../npyN2F_AgglomerativeClustering/',
        '../npyN2F_Birch/',
        '../npyN2F_KMeans/',
        '../npyN2F_SpectralClustering/',
        '../npyN2F/'
        ]

reguDict={2:None, 3:None}
for i in range(16,28):
    reguDict[i]=None
reguStr=''
if args.methodName in reguDict:
    reguStr=' --regulized-type noregu '

npyStr = npyList[args.methodName]

benchmarkStr = ''

if args.runMode:
    labelFileDir = '/home/wangjue/biodata/scData/AnjunBenchmark/'
else:
    labelFileDir = '/home/jwang/data/scData/'
    
def getBenchmarkStr(count):
    benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
                    '--n-clusters 3 '
    # benchmarkStr = ''
    # if int(count/3)==1:
    #     benchmarkStr = ' --benchmark '\
    #         '--labelFilename ' + labelFileDir + '4.Yan/Yan_cell_label.csv '\
    #         '--n-clusters 7 '
    # elif int(count/3)==2:
    #     benchmarkStr = ' --benchmark '\
    #         '--labelFilename ' + labelFileDir + '5.Goolam/Goolam_cell_label.csv '\
    #         '--n-clusters 5 '    
    # if not args.splitMode: 
    #     if int(count/3)==3:
    #         benchmarkStr = ' --benchmark '\
    #             '--labelFilename ' + labelFileDir + '7.Deng/Deng_cell_label.csv '\
    #             '--n-clusters 10 '   
    #     elif int(count/3)==4:
    #         benchmarkStr = ' --benchmark '\
    #             '--labelFilename ' + labelFileDir + '8.Pollen/Pollen_cell_label.csv '\
    #             '--n-clusters 11 '
    #     elif int(count/3)==5:
    #         benchmarkStr = ' --benchmark '\
    #             '--labelFilename ' + labelFileDir + '11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
    #             '--n-clusters 3 '
    # else:
    #     if not args.batchStr == 0:
    #         if int(count/3)==0:
    #             benchmarkStr = ' --benchmark '\
    #                 '--labelFilename ' + labelFileDir + '7.Deng/Deng_cell_label.csv '\
    #                 '--n-clusters 10 '
    #         elif int(count/3)==1:
    #             benchmarkStr = ' --benchmark '\
    #                 '--labelFilename ' + labelFileDir + '8.Pollen/Pollen_cell_label.csv '\
    #                 '--n-clusters 11 '
    #         elif int(count/3)==2:
    #             benchmarkStr = ' --benchmark '\
    #                 '--labelFilename ' + labelFileDir + '11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
    #                 '--n-clusters 3 '
    return benchmarkStr


if not args.runMode:
    if args.imputeMode:
        imputeStr = 'I'
    else:
        imputeStr = 'C'
    splitStr = ''
    if args.splitMode:
        splitStr = '_'+str(args.batchStr)+'_'
    templateStr = "#! /bin/bash\n"\
    "######################### Batch Headers #########################\n"\
    "#SBATCH -A xulab\n"\
    "#SBATCH -p BioCompute               # use the BioCompute partition\n"\
    "#SBATCH -J R" + imputeStr + splitStr + str(args.methodName) +              " \n"\
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
    commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + getBenchmarkStr(count) + ' --npyDir ' + npyStr
    if args.runMode:
        os.system(commandStr)
    else:
        print(commandStr)
    for i in range(5):
        commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + getBenchmarkStr(count) + ' --reconstr '+ str(i) + ' --npyDir ' + npyStr
        if args.runMode:
            os.system(commandStr)
        else:
            print(commandStr)
    count += 1


