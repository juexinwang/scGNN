import os
import argparse
parser = argparse.ArgumentParser(description='Read Results in different methods')
parser.add_argument('--methodName', type=int, default=0, 
                    help="method used: 0-62")
parser.add_argument('--imputeMode', default=True, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
parser.add_argument('--runMode',action='store_true', default=False, help="Run or prepare cluster script")
parser.add_argument('--splitMode', default=False, action='store_true',
                    help='whether split, used for long queue')
parser.add_argument('--batchStr', type=int, default=0, 
                    help="method used: 1-13")
args = parser.parse_args()

# Note:
# Generate results in python other than in shell for better organization
# We are not use runpy.run_path('main_result.py') for it is hard to pass arguments
# We are not use subprocess.call("python main_result.py", shell=True) for it runs scripts parallel
# So we use os.system('') here

if args.splitMode:
    #The split of batch, more batches, more parallel

    if args.batchStr == 8:
        datasetList = [
        '9.Chung',
        # '9.Chung --discreteTag'
        ]
    elif args.batchStr == 11:
        datasetList = [
        '11.Kolodziejczyk',
        # '11.Kolodziejczyk --discreteTag'
        ]
    elif args.batchStr == 12:
        datasetList = [
        '12.Klein',
        # '12.Klein --discreteTag'
        ]
    elif args.batchStr == 13:
        datasetList = [
        '13.Zeisel',
        # '13.Zeisel --discreteTag'
        ]
else:
    datasetList = [
        '9.Chung',
        '11.Kolodziejczyk',
        '12.Klein',
        '13.Zeisel',
    ]

if args.imputeMode:
    pyStr = 'results_impute.py'

    npyList = [
        '../npyImputeG2E_1/ --ratio 0.1', #1
        '../npyImputeG2E_1/ --ratio 0.3', #2
        '../npyImputeG2E_1/ --ratio 0.6', #3
        '../npyImputeG2E_1/ --ratio 0.8', #4
        '../npyImputeG2EL_1/ --ratio 0.1', #5
        '../npyImputeG2EL_1/ --ratio 0.3', #6
        '../npyImputeG2EL_1/ --ratio 0.6', #7
        '../npyImputeG2EL_1/ --ratio 0.8', #8
        '../npyImputeG1E_1/ --ratio 0.1', #9
        '../npyImputeG1E_1/ --ratio 0.3', #10
        '../npyImputeG1E_1/ --ratio 0.6', #11
        '../npyImputeG1E_1/ --ratio 0.8', #12
        '../npyImputeG2F_1/ --ratio 0.1', #13
        '../npyImputeG2F_1/ --ratio 0.3', #14
        '../npyImputeG2F_1/ --ratio 0.6', #15
        '../npyImputeG2F_1/ --ratio 0.8', #16
        '../npyImputeN2E_1/ --ratio 0.1', #17
        '../npyImputeN2E_1/ --ratio 0.3', #18
        '../npyImputeN2E_1/ --ratio 0.6', #19
        '../npyImputeN2E_1/ --ratio 0.8', #20

        '../npyImputeG2E_2/ --ratio 0.1', #21
        '../npyImputeG2E_2/ --ratio 0.3', #22
        '../npyImputeG2E_2/ --ratio 0.6', #23
        '../npyImputeG2E_2/ --ratio 0.8', #24
        '../npyImputeG2EL_2/ --ratio 0.1', #25
        '../npyImputeG2EL_2/ --ratio 0.3', #26
        '../npyImputeG2EL_2/ --ratio 0.6', #27
        '../npyImputeG2EL_2/ --ratio 0.8', #28
        '../npyImputeG1E_2/ --ratio 0.1', #29
        '../npyImputeG1E_2/ --ratio 0.3', #30
        '../npyImputeG1E_2/ --ratio 0.6', #31
        '../npyImputeG1E_2/ --ratio 0.8', #32
        '../npyImputeG2F_2/ --ratio 0.1', #33
        '../npyImputeG2F_2/ --ratio 0.3', #34
        '../npyImputeG2F_2/ --ratio 0.6', #35
        '../npyImputeG2F_2/ --ratio 0.8', #36
        '../npyImputeN2E_2/ --ratio 0.1', #37
        '../npyImputeN2E_2/ --ratio 0.3', #38
        '../npyImputeN2E_2/ --ratio 0.6', #39
        '../npyImputeN2E_2/ --ratio 0.8', #40

        '../npyImputeG2E_3/ --ratio 0.1', #41
        '../npyImputeG2E_3/ --ratio 0.3', #42
        '../npyImputeG2E_3/ --ratio 0.6', #43
        '../npyImputeG2E_3/ --ratio 0.8', #44
        '../npyImputeG2EL_3/ --ratio 0.1', #45
        '../npyImputeG2EL_3/ --ratio 0.3', #46
        '../npyImputeG2EL_3/ --ratio 0.6', #47
        '../npyImputeG2EL_3/ --ratio 0.8', #48
        '../npyImputeG1E_3/ --ratio 0.1', #49
        '../npyImputeG1E_3/ --ratio 0.3', #50
        '../npyImputeG1E_3/ --ratio 0.6', #51
        '../npyImputeG1E_3/ --ratio 0.8', #52
        '../npyImputeG2F_3/ --ratio 0.1', #53
        '../npyImputeG2F_3/ --ratio 0.3', #54
        '../npyImputeG2F_3/ --ratio 0.6', #55
        '../npyImputeG2F_3/ --ratio 0.8', #56
        '../npyImputeN2E_3/ --ratio 0.1', #57
        '../npyImputeN2E_3/ --ratio 0.3', #58
        '../npyImputeN2E_3/ --ratio 0.6', #59
        '../npyImputeN2E_3/ --ratio 0.8', #60

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

for i in range(0,16):
    reguDict[i]='LTMG'
for i in range(16,20):
    reguDict[i]='noregu'
for i in range(20,36):
    reguDict[i]='LTMG'
for i in range(36,40):
    reguDict[i]='noregu'
for i in range(40,56):
    reguDict[i]='LTMG'
for i in range(56,60):
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


