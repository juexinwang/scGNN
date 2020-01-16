import os
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods, for All version of MMP')
parser.add_argument('--methodName', type=int, default=0, 
                    help="method used: 0-27")
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
args = parser.parse_args()

# Note:
# Generate results in python other than in shell for better organization
# We are not use runpy.run_path('main_result.py') for it is hard to pass arguments
# We are not use subprocess.call("python main_result.py", shell=True) for it runs scripts parallel
# So we use os.system('') here
# Method 0-27 are list as npyList
#
datasetList = [
    'MMPbasal',
    'MMPbasal --discreteTag',
    'MMPbasal_LTMG',
    'MMPbasal_all',
    'MMPbasal_all --discreteTag',
    'MMPbasal_all_LTMG',
    'MMPbasal_allcell',
    'MMPbasal_allcell --discreteTag',
    'MMPbasal_2000',
    'MMPbasal_2000 --discreteTag',
    'MMPbasal_2000_LTMG'
    ]
    #TODO: we wait for 11.Kolodziejczyk_LTMG

if args.imputeMode:
    pyStr = 'results_impute.py'
    npyList = [
        '../npyImputeG1E/',
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
        '../npyImputeN2F/'
        ]
else:
    pyStr = 'results_celltype.py'
    npyList = [
        '../npyG1E/',
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
        '../npyN2F/'
        ]

reguDict={2:None, 3:None}
for i in range(16,28):
    reguDict[i]=None
reguStr=''
if args.methodName in reguDict:
    reguStr=' --regulized-type noregu '

npyStr = npyList[args.methodName]

for datasetStr in datasetList:
    commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + ' --npyDir ' + npyStr
    os.system(commandStr)
    # print(commandStr)
    for i in range(5):
        commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + ' --reconstr '+ str(i) + ' --npyDir ' + npyStr
        os.system(commandStr)
        # print(commandStr)

