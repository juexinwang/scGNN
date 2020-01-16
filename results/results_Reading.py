import os
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods')
parser.add_argument('--methodName', type=int, default=0, 
                    help="method used: 0-27")
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
parser.add_argument('--benchmark',action='store_true', default=False, help="whether have benchmark")
args = parser.parse_args()

# Note:
# Generate results in python other than in shell for better organization
# We are not use runpy.run_path('main_result.py') for it is hard to pass arguments
# We are not use subprocess.call("python main_result.py", shell=True) for it runs scripts parallel
# So we use os.system('') here
datasetList = ['MMPbasal_2000',
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
    '11_Kolodziejczyk --discreteTag']
    #TODO: we wait for 11.Kolodziejczyk_LTMG

if args.imputeMode:
    pyStr = 'results_impute.py'
    npyList = ['../npyImputeG1E/',
        '../npyImputeG1F/',
        '../npyImputeN1E/',
        '../npyImputeN1F/',
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
        '../npyImputeN2F/']
else:
    pyStr = 'results_celltype.py'
    npyList = ['../npyG1E/',
        '../npyG1F/',
        '../npyN1E/',
        '../npyN1F/',
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
        '../npyN2F/']

reguDict={2:None, 3:None}
for i in range(16,28):
    reguDict[i]=None
reguStr=''
if args.methodName in reguDict:
    reguStr=' --regulized-type noregu '

npyStr = npyList[args.methodName]

benchmarkStr = ''
if args.benchmark:
    if int(args.methodName/3)==1:
        benchmarkStr = ' --benchmark '\
            '--labelFilename /home/wangjue/biodata/scData/AnjunBenchmark/4.Yan/4.Yan_cell_label.csv '\
            '--cellFilename /home/wangjue/biodata/scData/4.Yan.cellname.txt '\
            '--cellIndexname /home/wangjue/myprojects/scGNN/data/sc/4.Yan/ind.4.Yan.cellindex.txt '
    elif int(args.methodName/3)==2:
        benchmarkStr = ' --benchmark '\
            '--labelFilename /home/wangjue/biodata/scData/AnjunBenchmark/5.Goolam/5.Goolam_cell_label.csv '\
            '--cellFilename /home/wangjue/biodata/scData/5.Goolam.cellname.txt '\
            '--cellIndexname /home/wangjue/myprojects/scGNN/data/sc/5.Goolam/ind.5.Goolam.cellindex.txt '
    elif int(args.methodName/3)==3:
        benchmarkStr = ' --benchmark '\
            '--labelFilename /home/wangjue/biodata/scData/AnjunBenchmark/7.Deng/7.Deng_cell_label.csv '\
            '--cellFilename /home/wangjue/biodata/scData/7.Deng.cellname.txt '\
            '--cellIndexname /home/wangjue/myprojects/scGNN/data/sc/7.Deng/ind.7.Deng.cellindex.txt '
    elif int(args.methodName/3)==4:
        benchmarkStr = ' --benchmark '\
            '--labelFilename /home/wangjue/biodata/scData/AnjunBenchmark/8.Pollen/8.Pollen_cell_label.csv '\
            '--cellFilename /home/wangjue/biodata/scData/8.Pollen.cellname.txt '\
            '--cellIndexname /home/wangjue/myprojects/scGNN/data/sc/8.Pollen/ind.8.Pollen.cellindex.txt '
    elif int(args.methodName/3)==5:
        benchmarkStr = ' --benchmark '\
            '--labelFilename /home/wangjue/biodata/scData/AnjunBenchmark/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
            '--cellFilename /home/wangjue/biodata/scData/11.Kolodziejczyk.cellname.txt '\
            '--cellIndexname /home/wangjue/myprojects/scGNN/data/sc/11.Kolodziejczyk/ind.11.Kolodziejczyk.cellindex.txt '


for datasetStr in datasetList:
    commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + benchmarkStr + ' --npyDir ' + npyStr
    # os.system(commandStr)
    print(commandStr)
    for i in range(5):
        commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + benchmarkStr + ' --reconstr '+ str(i) + ' --npyDir ' + npyStr
        # os.system(commandStr)
        print(commandStr)

