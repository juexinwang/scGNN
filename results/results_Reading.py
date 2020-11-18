import os
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods')
parser.add_argument('--methodName', type=int, default=0, 
                    help="method used: 0-62")
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
parser.add_argument('--runMode',action='store_true', default=False, help="Run or prepare cluster script")
parser.add_argument('--splitMode', default=False, action='store_true',
                    help='whether split, used for long queue')
parser.add_argument('--batchStr', type=int, default=0, 
                    help="method used: 0-12")
args = parser.parse_args()

# Note:
# Main Check results
# Generate results in python other than in shell for better organization
# We are not use runpy.run_path('main_result.py') for it is hard to pass arguments
# We are not use subprocess.call("python main_result.py", shell=True) for it runs scripts parallel
# So we use os.system('') here

if args.splitMode:
    #The split of batch, more batches, more parallel

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

    if args.batchStr == 0:
        datasetList = [
        '1.Biase',
        # '1.Biase --discreteTag'
        ]
    elif args.batchStr == 1:
        datasetList = [
        '2.Li',
        # '2.Li --discreteTag'
        ]
    elif args.batchStr == 2:
        datasetList = [
        '3.Treutlein',
        # '3.Treutlein --discreteTag'
        ]
    elif args.batchStr == 3:
        datasetList = [
        '4.Yan',
        # '4.Yan --discreteTag'
        ]
    elif args.batchStr == 4:
        datasetList = [
        '5.Goolam',
        # '5.Goolam --discreteTag'
        ]
    elif args.batchStr == 5:
        datasetList = [
        '6.Guo',
        # '6.Guo --discreteTag'
        ]
    elif args.batchStr == 6:
        datasetList = [
        '7.Deng',
        # '7.Deng --discreteTag'
        ]
    elif args.batchStr == 7:
        datasetList = [
        '8.Pollen',
        # '8.Pollen --discreteTag'
        ]
    elif args.batchStr == 8:
        datasetList = [
        '9.Chung',
        # '9.Chung --discreteTag'
        ]
    elif args.batchStr == 9:
        datasetList = [
        '10.Usoskin',
        # '10.Usoskin --discreteTag'
        ]
    elif args.batchStr == 10:
        datasetList = [
        '11.Kolodziejczyk',
        # '11.Kolodziejczyk --discreteTag'
        ]
    elif args.batchStr == 11:
        datasetList = [
        '12.Klein',
        # '12.Klein --discreteTag'
        ]
    elif args.batchStr == 12:
        datasetList = [
        '13.Zeisel',
        # '13.Zeisel --discreteTag'
        ]
    # elif args.batchStr == 13:
    #     datasetList = [
    #     '20.10X_2700_seurat',
    #     # '20.10X_2700_seurat --discreteTag'
    #     ]
    # elif args.batchStr == 14:
    #     datasetList = [
    #     '30.Schafer',
    #     # '30.Schafer --discreteTag'
    #     ]
else:
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
        '20.10X_2700_seurat',
        '20.10X_2700_seurat --discreteTag',
        '30.Schafer',
        '30.Schafer --discreteTag'
    ]

if args.imputeMode:
    pyStr = 'results_impute.py'
    #complex
    # npyList = [
    #     '../npyImputeG1B/', #0
    #     '../npyImputeG1E/', #1
    #     '../npyImputeG1F/', #2
    #     '../npyImputeR1B/', #3
    #     '../npyImputeR1E/', #4
    #     '../npyImputeR1F/', #5
    #     '../npyImputeN1B/', #6
    #     '../npyImputeN1E/', #7
    #     '../npyImputeN1F/', #8
    #     '../npyImputeG2B_AffinityPropagation/',  #9
    #     '../npyImputeG2B_AgglomerativeClustering/', #10
    #     '../npyImputeG2B_Birch/', #11
    #     '../npyImputeG2B_KMeans/',  #12
    #     '../npyImputeG2B_SpectralClustering/', #13
    #     '../npyImputeG2B/', #14
    #     '../npyImputeG2E_AffinityPropagation/', #15
    #     '../npyImputeG2E_AgglomerativeClustering/', #16
    #     '../npyImputeG2E_Birch/', #17
    #     '../npyImputeG2E_KMeans/', #18
    #     '../npyImputeG2E_SpectralClustering/', #19
    #     '../npyImputeG2E/', #20
    #     '../npyImputeG2F_AffinityPropagation/', #21
    #     '../npyImputeG2F_AgglomerativeClustering/', #22
    #     '../npyImputeG2F_Birch/', #23
    #     '../npyImputeG2F_KMeans/', #24
    #     '../npyImputeG2F_SpectralClustering/', #25
    #     '../npyImputeG2F/', #26
    #     '../npyImputeR2B_AffinityPropagation/',  #27
    #     '../npyImputeR2B_AgglomerativeClustering/', #28
    #     '../npyImputeR2B_Birch/', #29
    #     '../npyImputeR2B_KMeans/', #30
    #     '../npyImputeR2B_SpectralClustering/', #31
    #     '../npyImputeR2B/', #32
    #     '../npyImputeR2E_AffinityPropagation/', #33
    #     '../npyImputeR2E_AgglomerativeClustering/', #34
    #     '../npyImputeR2E_Birch/', #35
    #     '../npyImputeR2E_KMeans/', #36
    #     '../npyImputeR2E_SpectralClustering/', #37
    #     '../npyImputeR2E/', #38
    #     '../npyImputeR2F_AffinityPropagation/', #39
    #     '../npyImputeR2F_AgglomerativeClustering/', #40
    #     '../npyImputeR2F_Birch/', #41
    #     '../npyImputeR2F_KMeans/', #42
    #     '../npyImputeR2F_SpectralClustering/', #43
    #     '../npyImputeR2F/', #44
    #     '../npyImputeN2B_AffinityPropagation/', #45
    #     '../npyImputeN2B_AgglomerativeClustering/', #46
    #     '../npyImputeN2B_Birch/', #47
    #     '../npyImputeN2B_KMeans/', #48
    #     '../npyImputeN2B_SpectralClustering/', #49
    #     '../npyImputeN2B/', #50
    #     '../npyImputeN2E_AffinityPropagation/', #51
    #     '../npyImputeN2E_AgglomerativeClustering/', #52
    #     '../npyImputeN2E_Birch/', #53
    #     '../npyImputeN2E_KMeans/', #54
    #     '../npyImputeN2E_SpectralClustering/', #55
    #     '../npyImputeN2E/', #56
    #     '../npyImputeN2F_AffinityPropagation/', #57
    #     '../npyImputeN2F_AgglomerativeClustering/', #58
    #     '../npyImputeN2F_Birch/',  #59
    #     '../npyImputeN2F_KMeans/',  #60
    #     '../npyImputeN2F_SpectralClustering/', #61
    #     '../npyImputeN2F/'  #62
    #     ]

    # npyList = [
    #     '../npyImputeG1B/', #0
    #     '../npyImputeG1E/', #1
    #     '../npyImputeR1B/', #2
    #     '../npyImputeR1E/', #3
    #     '../npyImputeG2B/', #4
    #     '../npyImputeG2E/', #5
    #     '../npyImputeR2B/', #6
    #     '../npyImputeR2E/', #7
    #     '../npyImputeG2B_Birch/', #8
    #     '../npyImputeG2B_BirchN/', #9
    #     '../npyImputeG2B_KMeans/', #10
    #     '../npyImputeG2E_Birch/', #11
    #     '../npyImputeG2E_BirchN/', #12
    #     '../npyImputeG2E_KMeans/', #13
    #     '../npyImputeR2B_Birch/', #14
    #     '../npyImputeR2B_BirchN/', #15
    #     '../npyImputeR2B_KMeans/', #16
    #     '../npyImputeR2E_Birch/', #17
    #     '../npyImputeR2E_BirchN/', #18
    #     '../npyImputeR2E_KMeans/', #19
    #     ]

    # npyList = [
    #     '../npyImputeG1B/', #0
    #     '../npyImputeG1E/', #1
    #     '../npyImputeR1B/', #2
    #     '../npyImputeR1E/', #3
    #     '../npyImputeG2B/', #4
    #     '../npyImputeG2E/', #5
    #     '../npyImputeR2B/', #6
    #     '../npyImputeR2E/', #7
    #     '../npyImputeG2B_Birch/', #8
    #     '../npyImputeG2B_KMeans/', #9
    #     '../npyImputeG2E_Birch/', #10
    #     '../npyImputeG2E_KMeans/', #11
    #     '../npyImputeR2B_Birch/', #12
    #     '../npyImputeR2B_KMeans/', #13
    #     '../npyImputeR2E_Birch/', #14
    #     '../npyImputeR2E_KMeans/', #15
    #     ]

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
        '../npyImputeG2B/', #9
        '../npyImputeG2E/', #10
        '../npyImputeG2F/', #11
        '../npyImputeR2B/', #12
        '../npyImputeR2E/', #13
        '../npyImputeR2F/', #14
        '../npyImputeN2B/', #15
        '../npyImputeN2E/', #16
        '../npyImputeN2F/', #17

        '../npyImputeG1B_LK/', #18
        '../npyImputeG1E_LK/', #19
        '../npyImputeG1F_LK/', #20
        '../npyImputeR1B_LK/', #21
        '../npyImputeR1E_LK/', #22
        '../npyImputeR1F_LK/', #23
        '../npyImputeN1B_LK/', #24
        '../npyImputeN1E_LK/', #25
        '../npyImputeN1F_LK/', #26
        '../npyImputeG2B_LK/', #27
        '../npyImputeG2E_LK/', #28
        '../npyImputeG2F_LK/', #29
        '../npyImputeR2B_LK/', #30
        '../npyImputeR2E_LK/', #31
        '../npyImputeR2F_LK/', #32
        '../npyImputeN2B_LK/', #33
        '../npyImputeN2E_LK/', #34
        '../npyImputeN2F_LK/', #35

        '../npyImputeG1B_LB/', #36
        '../npyImputeG1E_LB/', #37
        '../npyImputeG1F_LB/', #38
        '../npyImputeR1B_LB/', #39
        '../npyImputeR1E_LB/', #40
        '../npyImputeR1F_LB/', #41
        '../npyImputeN1B_LB/', #42
        '../npyImputeN1E_LB/', #43
        '../npyImputeN1F_LB/', #44
        '../npyImputeG2B_LB/', #45
        '../npyImputeG2E_LB/', #46
        '../npyImputeG2F_LB/', #47
        '../npyImputeR2B_LB/', #48
        '../npyImputeR2E_LB/', #49
        '../npyImputeR2F_LB/', #50
        '../npyImputeN2B_LB/', #51
        '../npyImputeN2E_LB/', #52
        '../npyImputeN2F_LB/', #53
        ]

else:
    pyStr = 'results_celltype.py'
    # complex
    # npyList = [
    #     '../npyG1B/', #0
    #     '../npyG1E/', #1
    #     '../npyG1F/', #2
    #     '../npyR1B/', #3
    #     '../npyR1E/', #4
    #     '../npyR1F/', #5
    #     '../npyN1B/', #6
    #     '../npyN1E/', #7
    #     '../npyN1F/', #8
    #     '../npyG2B_AffinityPropagation/', #9
    #     '../npyG2B_AgglomerativeClustering/', #10
    #     '../npyG2B_Birch/', #11
    #     '../npyG2B_KMeans/', #12
    #     '../npyG2B_SpectralClustering/', #13
    #     '../npyG2B/', #14
    #     '../npyG2E_AffinityPropagation/', #15
    #     '../npyG2E_AgglomerativeClustering/', #16
    #     '../npyG2E_Birch/', #17
    #     '../npyG2E_KMeans/', #18
    #     '../npyG2E_SpectralClustering/', #19
    #     '../npyG2E/', #20
    #     '../npyG2F_AffinityPropagation/', #21
    #     '../npyG2F_AgglomerativeClustering/', #22
    #     '../npyG2F_Birch/', #23
    #     '../npyG2F_KMeans/', #24
    #     '../npyG2F_SpectralClustering/', #25
    #     '../npyG2F/', #26
    #     '../npyR2B_AffinityPropagation/', #27
    #     '../npyR2B_AgglomerativeClustering/', #28
    #     '../npyR2B_Birch/', #29
    #     '../npyR2B_KMeans/', #30
    #     '../npyR2B_SpectralClustering/', #31
    #     '../npyR2B/', #32
    #     '../npyR2E_AffinityPropagation/', #33
    #     '../npyR2E_AgglomerativeClustering/', #34
    #     '../npyR2E_Birch/', #35
    #     '../npyR2E_KMeans/', #36
    #     '../npyR2E_SpectralClustering/', #37
    #     '../npyR2E/', #38
    #     '../npyR2F_AffinityPropagation/', #39
    #     '../npyR2F_AgglomerativeClustering/', #40
    #     '../npyR2F_Birch/', #41
    #     '../npyR2F_KMeans/', #42
    #     '../npyR2F_SpectralClustering/', #43
    #     '../npyR2F/', #44
    #     '../npyN2B_AffinityPropagation/', #45
    #     '../npyN2B_AgglomerativeClustering/', #46
    #     '../npyN2B_Birch/', #47
    #     '../npyN2B_KMeans/', #48
    #     '../npyN2E_SpectralClustering/', #49
    #     '../npyN2E/', #50
    #     '../npyN2E_AffinityPropagation/', #51
    #     '../npyN2E_AgglomerativeClustering/', #52
    #     '../npyN2E_Birch/', #53
    #     '../npyN2E_KMeans/', #54
    #     '../npyN2E_SpectralClustering/', #55
    #     '../npyN2E/', #56
    #     '../npyN2F_AffinityPropagation/', #57
    #     '../npyN2F_AgglomerativeClustering/', #58
    #     '../npyN2F_Birch/', #59
    #     '../npyN2F_KMeans/', #60
    #     '../npyN2F_SpectralClustering/', #61
    #     '../npyN2F/'  #62
    #     ]

    # npyList = [
    #     '../npyG1B/', #0
    #     '../npyG1E/', #1
    #     '../npyR1B/', #2
    #     '../npyR1E/', #3
    #     '../npyG2B/', #4
    #     '../npyG2E/', #5
    #     '../npyR2B/', #6
    #     '../npyR2E/', #7
    #     '../npyG2B_Birch/', #8
    #     '../npyG2B_BirchN/', #9
    #     '../npyG2B_KMeans/', #10
    #     '../npyG2E_Birch/', #11
    #     '../npyG2E_BirchN/', #12
    #     '../npyG2E_KMeans/', #13
    #     '../npyR2B_Birch/', #14
    #     '../npyR2B_BirchN/', #15
    #     '../npyR2B_KMeans/', #16
    #     '../npyR2E_Birch/', #17
    #     '../npyR2E_BirchN/', #18
    #     '../npyR2E_KMeans/', #19
    #     ]
    
    # npyList = [
    #     '../npyG1B/', #0
    #     '../npyG1E/', #1
    #     '../npyR1B/', #2
    #     '../npyR1E/', #3
    #     '../npyG2B/', #4
    #     '../npyG2E/', #5
    #     '../npyR2B/', #6
    #     '../npyR2E/', #7
    #     '../npyG2B_Birch/', #8
    #     '../npyG2B_KMeans/', #9
    #     '../npyG2E_Birch/', #10
    #     '../npyG2E_KMeans/', #11
    #     '../npyR2B_Birch/', #12
    #     '../npyR2B_KMeans/', #13
    #     '../npyR2E_Birch/', #14
    #     '../npyR2E_KMeans/', #15
    #     ]

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
        '../npyG2B/', #9
        '../npyG2E/', #10
        '../npyG2F/', #11
        '../npyR2B/', #12
        '../npyR2E/', #13
        '../npyR2F/', #14
        '../npyN2B/', #15
        '../npyN2E/', #16
        '../npyN2F/', #17

        '../npyG1B_LK/', #18
        '../npyG1E_LK/', #19
        '../npyG1F_LK/', #20
        '../npyR1B_LK/', #21
        '../npyR1E_LK/', #22
        '../npyR1F_LK/', #23
        '../npyN1B_LK/', #24
        '../npyN1E_LK/', #25
        '../npyN1F_LK/', #26
        '../npyG2B_LK/', #27
        '../npyG2E_LK/', #28
        '../npyG2F_LK/', #29
        '../npyR2B_LK/', #30
        '../npyR2E_LK/', #31
        '../npyR2F_LK/', #32
        '../npyN2B_LK/', #33
        '../npyN2E_LK/', #34
        '../npyN2F_LK/', #35

        '../npyG1B_LB/', #36
        '../npyG1E_LB/', #37
        '../npyG1F_LB/', #38
        '../npyR1B_LB/', #39
        '../npyR1E_LB/', #40
        '../npyR1F_LB/', #41
        '../npyN1B_LB/', #42
        '../npyN1E_LB/', #43
        '../npyN1F_LB/', #44
        '../npyG2B_LB/', #45
        '../npyG2E_LB/', #46
        '../npyG2F_LB/', #47
        '../npyR2B_LB/', #48
        '../npyR2E_LB/', #49
        '../npyR2F_LB/', #50
        '../npyN2B_LB/', #51
        '../npyN2E_LB/', #52
        '../npyN2F_LB/', #53
        ]

reguDict={}
#complex
# for i in range(0,3):
#     reguDict[i]='LTMG'
# for i in range(3,6):
#     reguDict[i]='LTMG01'
# for i in range(6,9):
#     reguDict[i]='noregu'
# for i in range(9,27):
#     reguDict[i]='LTMG'
# for i in range(27,45):
#     reguDict[i]='LTMG01'
# for i in range(45,63):
#     reguDict[i]='noregu'

#select:
# for i in range(0,2):
#     reguDict[i]='LTMG'
# for i in range(2,4):
#     reguDict[i]='LTMG01'
# for i in range(4,6):
#     reguDict[i]='LTMG'
# for i in range(6,8):
#     reguDict[i]='LTMG01'
# for i in range(8,14):
#     reguDict[i]='LTMG'
# for i in range(14,20):
#     reguDict[i]='LTMG01'

#strong
# for i in range(0,2):
#     reguDict[i]='LTMG'
# for i in range(2,4):
#     reguDict[i]='LTMG01'
# for i in range(4,6):
#     reguDict[i]='LTMG'
# for i in range(6,8):
#     reguDict[i]='LTMG01'
# for i in range(8,12):
#     reguDict[i]='LTMG'
# for i in range(12,16):
#     reguDict[i]='LTMG01'

for i in range(0,3):
    reguDict[i]='LTMG'
for i in range(3,6):
    reguDict[i]='LTMG01'
for i in range(6,9):
    reguDict[i]='noregu'
for i in range(10,12):
    reguDict[i]='LTMG'
for i in range(12,15):
    reguDict[i]='LTMG01'
for i in range(15,18):
    reguDict[i]='noregu'

for i in range(18,21):
    reguDict[i]='LTMG'
for i in range(21,24):
    reguDict[i]='LTMG01'
for i in range(24,27):
    reguDict[i]='noregu'
for i in range(27,30):
    reguDict[i]='LTMG'
for i in range(30,33):
    reguDict[i]='LTMG01'
for i in range(33,36):
    reguDict[i]='noregu'

for i in range(36,39):
    reguDict[i]='LTMG'
for i in range(39,42):
    reguDict[i]='LTMG01'
for i in range(42,45):
    reguDict[i]='noregu'
for i in range(45,48):
    reguDict[i]='LTMG'
for i in range(48,51):
    reguDict[i]='LTMG01'
for i in range(51,54):
    reguDict[i]='noregu'

reguStr=''
if args.methodName in reguDict:
    reguStr=' --regulized-type ' + reguDict[args.methodName] + ' '

npyStr = npyList[args.methodName]

benchmarkStr = ''

if args.runMode:
    labelFileDir = '/home/wangjue/biodata/scData/allBench/'
else:
    labelFileDir = '/home/jwang/data/scData/'
    
def getBenchmarkStr(count):
    benchmarkStr = ''
    if args.batchStr == 0:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '1.Biase/Biase_cell_label.csv '\
                    '--n-clusters 3 '
    elif args.batchStr == 1:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '2.Li/Li_cell_label.csv '\
                    '--n-clusters 9 '
    elif args.batchStr == 2:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '3.Treutlein/Treutlein_cell_label.csv '\
                    '--n-clusters 5 '
    elif args.batchStr == 3:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '4.Yan/Yan_cell_label.csv '\
                    '--n-clusters 7 '
    elif args.batchStr == 4:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '5.Goolam/Goolam_cell_label.csv '\
                    '--n-clusters 5 '
    elif args.batchStr == 5:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '6.Guo/Guo_cell_label.csv '\
                    '--n-clusters 9 '
    elif args.batchStr == 6:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '7.Deng/Deng_cell_label.csv '\
                    '--n-clusters 10 '
    elif args.batchStr == 7:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '8.Pollen/Pollen_cell_label.csv '\
                    '--n-clusters 11 '
    elif args.batchStr == 8:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '9.Chung/Chung_cell_label.csv '\
                    '--n-clusters 4 '
    elif args.batchStr == 9:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '10.Usoskin/Usoskin_cell_label.csv '\
                    '--n-clusters 11 '
    elif args.batchStr == 10:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '11.Kolodziejczyk/Kolodziejczyk_cell_label.csv '\
                    '--n-clusters 3 '
    elif args.batchStr == 11:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '12.Klein/Klein_cell_label.csv '\
                    '--n-clusters 4 '
    elif args.batchStr == 12:
        benchmarkStr = ' --benchmark '\
                    '--labelFilename ' + labelFileDir + '13.Zeisel/Zeisel_cell_label.csv '\
                    '--n-clusters 7 '
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
        splitStr = '_'+str(args.batchStr)
    templateStr = "#! /bin/bash\n"\
    "######################### Batch Headers #########################\n"\
    "#SBATCH -A xulab\n"\
    "#SBATCH -p Lewis,BioCompute               # use the BioCompute partition\n"\
    "#SBATCH -J R" + imputeStr + '_' + str(args.methodName) + splitStr +              " \n"\
    "#SBATCH -o results-%j.out           # give the job output a custom name\n"\
    "#SBATCH -t 2-00:00                  # two days time limit\n"\
    "#SBATCH -N 1                        # number of nodes\n"\
    "#SBATCH -n 1                        # number of cores (AKA tasks)\n"\
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
    for i in range(10):
        commandStr = 'python -W ignore ' + pyStr + ' --datasetName ' + datasetStr + reguStr + getBenchmarkStr(count) + ' --reconstr '+ str(i) + ' --npyDir ' + npyStr
        if args.runMode:
            os.system(commandStr)
        else:
            print(commandStr)
    count += 1


