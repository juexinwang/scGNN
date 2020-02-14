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

for datasetStr in datasetList:
    commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + ' --npyDir ' + npyStr
    os.system(commandStr)
    # print(commandStr)
    for i in range(5):
        commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + ' --reconstr '+ str(i) + ' --npyDir ' + npyStr
        os.system(commandStr)
        # print(commandStr)

