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

    # gradients
    # if args.batchStr == 0:
    #     datasetList = [
    #     'T1000',
    #     'T1000 --discreteTag',
    #     'T1000_LTMG'
    #     ]
    # elif args.batchStr == 1:
    #     datasetList = [
    #     'T2000',
    #     'T2000 --discreteTag',
    #     'T2000_LTMG'
    #     ]
    # elif args.batchStr == 2:
    #     datasetList = [
    #     'T4000',
    #     'T4000 --discreteTag',
    #     'T4000_LTMG'
    #     ]
    # elif args.batchStr == 3:
    #     datasetList = [
    #     'T8000',
    #     'T8000 --discreteTag',
    #     'T8000_LTMG'
    #     ]

    # gradients
    if args.batchStr == 0:
        datasetList = [
        '1.Biase'
        ]
    elif args.batchStr == 1:
        datasetList = [
        '2.Li'
        ]
    elif args.batchStr == 2:
        datasetList = [
        '3.Treutlein'
        ]
    elif args.batchStr == 3:
        datasetList = [
        '4.Yan'
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
        '../npyImputeR1B/', #3
        '../npyImputeR1E/', #4
        '../npyImputeR1F/', #5
        '../npyImputeN1B/', #6
        '../npyImputeN1E/', #7
        '../npyImputeN1F/', #8
        '../npyImputeG2B_AffinityPropagation/',  #9
        '../npyImputeG2B_AgglomerativeClustering/', #10
        '../npyImputeG2B_Birch/', #11
        '../npyImputeG2B_KMeans/',  #12
        '../npyImputeG2B_SpectralClustering/', #13
        '../npyImputeG2B/', #14
        '../npyImputeG2E_AffinityPropagation/', #15
        '../npyImputeG2E_AgglomerativeClustering/', #16
        '../npyImputeG2E_Birch/', #17
        '../npyImputeG2E_KMeans/', #18
        '../npyImputeG2E_SpectralClustering/', #19
        '../npyImputeG2E/', #20
        '../npyImputeG2F_AffinityPropagation/', #21
        '../npyImputeG2F_AgglomerativeClustering/', #22
        '../npyImputeG2F_Birch/', #23
        '../npyImputeG2F_KMeans/', #24
        '../npyImputeG2F_SpectralClustering/', #25
        '../npyImputeG2F/', #26
        '../npyImputeR2B_AffinityPropagation/',  #27
        '../npyImputeR2B_AgglomerativeClustering/', #28
        '../npyImputeR2B_Birch/', #29
        '../npyImputeR2B_KMeans/', #30
        '../npyImputeR2B_SpectralClustering/', #31
        '../npyImputeR2B/', #32
        '../npyImputeR2E_AffinityPropagation/', #33
        '../npyImputeR2E_AgglomerativeClustering/', #34
        '../npyImputeR2E_Birch/', #35
        '../npyImputeR2E_KMeans/', #36
        '../npyImputeR2E_SpectralClustering/', #37
        '../npyImputeR2E/', #38
        '../npyImputeR2F_AffinityPropagation/', #39
        '../npyImputeR2F_AgglomerativeClustering/', #40
        '../npyImputeR2F_Birch/', #41
        '../npyImputeR2F_KMeans/', #42
        '../npyImputeR2F_SpectralClustering/', #43
        '../npyImputeR2F/', #44
        '../npyImputeN2B_AffinityPropagation/', #45
        '../npyImputeN2B_AgglomerativeClustering/', #46
        '../npyImputeN2B_Birch/', #47
        '../npyImputeN2B_KMeans/', #48
        '../npyImputeN2B_SpectralClustering/', #49
        '../npyImputeN2B/', #50
        '../npyImputeN2E_AffinityPropagation/', #51
        '../npyImputeN2E_AgglomerativeClustering/', #52
        '../npyImputeN2E_Birch/', #53
        '../npyImputeN2E_KMeans/', #54
        '../npyImputeN2E_SpectralClustering/', #55
        '../npyImputeN2E/', #56
        '../npyImputeN2F_AffinityPropagation/', #57
        '../npyImputeN2F_AgglomerativeClustering/', #58
        '../npyImputeN2F_Birch/',  #59
        '../npyImputeN2F_KMeans/',  #60
        '../npyImputeN2F_SpectralClustering/', #61
        '../npyImputeN2F/'  #62
        ]
else:
    pyStr = 'results_celltype.py'
    npyList = [
        '../npyG1B/', #0
        '../npyG1E/', #1
        '../npyG1F/', #2
        '../npyR1B/', #3
        '../npyR1E/', #4
        '../npyR1F/', #5
        '../npyN1B/', #6
        '../npyN1E/', #7
        '../npyN1F/', #8
        '../npyG2B_AffinityPropagation/', #9
        '../npyG2B_AgglomerativeClustering/', #10
        '../npyG2B_Birch/', #11
        '../npyG2B_KMeans/', #12
        '../npyG2B_SpectralClustering/', #13
        '../npyG2B/', #14
        '../npyG2E_AffinityPropagation/', #15
        '../npyG2E_AgglomerativeClustering/', #16
        '../npyG2E_Birch/', #17
        '../npyG2E_KMeans/', #18
        '../npyG2E_SpectralClustering/', #19
        '../npyG2E/', #20
        '../npyG2F_AffinityPropagation/', #21
        '../npyG2F_AgglomerativeClustering/', #22
        '../npyG2F_Birch/', #23
        '../npyG2F_KMeans/', #24
        '../npyG2F_SpectralClustering/', #25
        '../npyG2F/', #26
        '../npyR2B_AffinityPropagation/', #27
        '../npyR2B_AgglomerativeClustering/', #28
        '../npyR2B_Birch/', #29
        '../npyR2B_KMeans/', #30
        '../npyR2B_SpectralClustering/', #31
        '../npyR2B/', #32
        '../npyR2E_AffinityPropagation/', #33
        '../npyR2E_AgglomerativeClustering/', #34
        '../npyR2E_Birch/', #35
        '../npyR2E_KMeans/', #36
        '../npyR2E_SpectralClustering/', #37
        '../npyR2E/', #38
        '../npyR2F_AffinityPropagation/', #39
        '../npyR2F_AgglomerativeClustering/', #40
        '../npyR2F_Birch/', #41
        '../npyR2F_KMeans/', #42
        '../npyR2F_SpectralClustering/', #43
        '../npyR2F/', #44
        '../npyN2B_AffinityPropagation/', #45
        '../npyN2B_AgglomerativeClustering/', #46
        '../npyN2B_Birch/', #47
        '../npyN2B_KMeans/', #48
        '../npyN2E_SpectralClustering/', #49
        '../npyN2E/', #50
        '../npyN2E_AffinityPropagation/', #51
        '../npyN2E_AgglomerativeClustering/', #52
        '../npyN2E_Birch/', #53
        '../npyN2E_KMeans/', #54
        '../npyN2E_SpectralClustering/', #55
        '../npyN2E/', #56
        '../npyN2F_AffinityPropagation/', #57
        '../npyN2F_AgglomerativeClustering/', #58
        '../npyN2F_Birch/', #59
        '../npyN2F_KMeans/', #60
        '../npyN2F_SpectralClustering/', #61
        '../npyN2F/'  #62
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
    "#SBATCH --mem=128G\n"\
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


