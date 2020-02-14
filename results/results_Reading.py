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
    # '4.Yan',
    # '4.Yan --discreteTag',
    # '5.Goolam',
    # '5.Goolam --discreteTag',
    # '7.Deng',
    # '7.Deng --discreteTag',
    # '8.Pollen',
    # '8.Pollen --discreteTag',
    # '11.Kolodziejczyk',
    # '11.Kolodziejczyk --discreteTag'
    ]

if args.imputeMode:
    pyStr = 'results_impute.py'
    npyList = [
        '../npyImputeG1B/', #0
        '../npyImputeG1E/', #1
        '../npyImputeG1F/', #2
        '../npyImputeN1B/', #3
        '../npyImputeN1E/', #4
        '../npyImputeN1F/', #5
        '../npyImputeG2B_AffinityPropagation/',  #6
        '../npyImputeG2B_AgglomerativeClustering/', #7
        '../npyImputeG2B_Birch/', #8
        '../npyImputeG2B_KMeans/',  #9
        '../npyImputeG2B_SpectralClustering/', #10
        '../npyImputeG2B/', #11
        '../npyImputeG2E_AffinityPropagation/', #12
        '../npyImputeG2E_AgglomerativeClustering/', #13
        '../npyImputeG2E_Birch/', #14
        '../npyImputeG2E_KMeans/', #15
        '../npyImputeG2E_SpectralClustering/', #16
        '../npyImputeG2E/', #17
        '../npyImputeG2F_AffinityPropagation/', #18
        '../npyImputeG2F_AgglomerativeClustering/', #19
        '../npyImputeG2F_Birch/', #20
        '../npyImputeG2F_KMeans/', #21
        '../npyImputeG2F_SpectralClustering/', #22
        '../npyImputeG2F/', #23
        '../npyImputeN2B_AffinityPropagation/', #24
        '../npyImputeN2B_AgglomerativeClustering/', #25
        '../npyImputeN2B_Birch/', #26
        '../npyImputeN2B_KMeans/', #27
        '../npyImputeN2B_SpectralClustering/', #28
        '../npyImputeN2B/', #29
        '../npyImputeN2E_AffinityPropagation/', #30
        '../npyImputeN2E_AgglomerativeClustering/', #31
        '../npyImputeN2E_Birch/', #32
        '../npyImputeN2E_KMeans/', #33
        '../npyImputeN2E_SpectralClustering/', #34
        '../npyImputeN2E/', #35
        '../npyImputeN2F_AffinityPropagation/', #36
        '../npyImputeN2F_AgglomerativeClustering/', #37
        '../npyImputeN2F_Birch/',  #38
        '../npyImputeN2F_KMeans/',  #39
        '../npyImputeN2F_SpectralClustering/', #40
        '../npyImputeN2F/'  #41
        ]
else:
    pyStr = 'results_celltype.py'
    npyList = [
        '../npyG1B/', #0
        '../npyG1E/', #1
        '../npyG1F/', #2
        '../npyN1B/', #3
        '../npyN1E/', #4
        '../npyN1F/', #5
        '../npyG2B_AffinityPropagation/', #6
        '../npyG2B_AgglomerativeClustering/', #7
        '../npyG2B_Birch/', #8
        '../npyG2B_KMeans/', #9
        '../npyG2B_SpectralClustering/', #10
        '../npyG2B/', #11
        '../npyG2E_AffinityPropagation/', #12
        '../npyG2E_AgglomerativeClustering/', #13
        '../npyG2E_Birch/', #14
        '../npyG2E_KMeans/', #15
        '../npyG2E_SpectralClustering/', #16
        '../npyG2E/', #17
        '../npyG2F_AffinityPropagation/', #18
        '../npyG2F_AgglomerativeClustering/', #19
        '../npyG2F_Birch/', #20
        '../npyG2F_KMeans/', #21
        '../npyG2F_SpectralClustering/', #22
        '../npyG2F/', #23
        '../npyN2B_AffinityPropagation/', #24
        '../npyN2B_AgglomerativeClustering/', #25
        '../npyN2B_Birch/', #26
        '../npyN2B_KMeans/', #27
        '../npyN2E_SpectralClustering/', #28
        '../npyN2E/', #29
        '../npyN2E_AffinityPropagation/', #30
        '../npyN2E_AgglomerativeClustering/', #31
        '../npyN2E_Birch/', #32
        '../npyN2E_KMeans/', #33
        '../npyN2E_SpectralClustering/', #34
        '../npyN2E/', #35
        '../npyN2F_AffinityPropagation/', #36
        '../npyN2F_AgglomerativeClustering/', #37
        '../npyN2F_Birch/', #38
        '../npyN2F_KMeans/', #39
        '../npyN2F_SpectralClustering/', #40
        '../npyN2F/'  #41
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


