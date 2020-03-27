import argparse

# python generatingCasestudy.py

parser = argparse.ArgumentParser(description='Generating sbatch files for HPC cluster running')
parser.add_argument('--outputDir', type=str, default='',
                    help='Directory of batch files for cluster running')
args = parser.parse_args()

templateStr1 = "#! /bin/bash\n"\
"######################### Batch Headers #########################\n"\
"#SBATCH -A xulab\n"\
"#SBATCH -p BioCompute               # use the BioCompute partition\n"\
"#SBATCH -J "

templateStr2 = "\n#SBATCH -o results-%j.out           # give the job output a custom name\n"\
"#SBATCH -t 2-00:00                  # two days time limit\n"\
"#SBATCH -N 1                        # number of nodes\n"\
"#SBATCH -n 2                        # number of cores (AKA tasks)\n"\
"#SBATCH --mem=128G\n"\
"#################################################################\n"\
"module load miniconda3\n"\
"source activate conda_R\n"

methodsList = [
    ('run_case_2geLK E1','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','casenpyG2E_LK_/'),
    ('run_case_2reLK E2','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','casenpyR2E_LK_/'),
    ('run_case_2geLB E3','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','casenpyG2E_LB_/'),
    ('run_case_2reLB E4','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','casenpyR2E_LB_/'),    
]

# select
datasetNameList = [
    'AD_GSE138852_2x8CT',  #1
    'AD_GSE103334_NORMED_8CT', #2
    'E-GEOD-139324', #3
    'liver_cancer_GSE98638_11CT', #4
    'liver_cancer_10X_20CT',  #5
    'liver_cancer_smart_22CT', #6
    'GSM2388072', #7
    'BR', #8
    'Par', #9
    '7f5e7a85-a45c-4535-90bf-0ef930a0040b.mtx', #10
    '481193cb-c021-4e04-b477-0b7cfef4614b.mtx', #11
    'e7448a34-b33d-41de-b422-4c09bfeba96b.mtx', #12: data1
    'c0b92850-ce85-41bb-b928-5fac1a113fef.mtx', #13: data2
]

reguParaList = [
    '--gammaPara 0.9 --regularizePara 0.1',
    '--gammaPara 0.5 --regularizePara 0.5',
    '--gammaPara 0.1 --regularizePara 0.9',
    ]

l12ParaList = [
    '--L1Para 0.0 --L2Para 0.0',
    '--L1Para 0.001 --L2Para 0.0',
    '--L1Para 0.0 --L2Para 0.001',
    '--L1Para 0.001 --L2Para 0.001',
    ]

# generate sbatch files:
for item in methodsList:
    batchInfo,scGNNparam,outDirStr = item
    tmp = batchInfo.split()
    tmpstr1=tmp[0]
    tmpstr2=tmp[1]
    outputFilename = args.outputDir + tmpstr1
    abbrStr = tmpstr2   

    count = 1
    for datasetName in datasetNameList:
        rcount = 1
        for rP in reguParaList:
            lcount = 1
            for lP in l12ParaList:
                commandLine = "python3 -W ignore scGNN.py --datasetName "+datasetName+" "+rP+" "+lP+" "+scGNNparam+" "+outDirStr+"\n"
                outStr = templateStr1 + abbrStr + "_" + str(count) + "_" + str(rcount) + "_" + str(lcount)+ templateStr2 + commandLine + "\n"
                with open(outputFilename+"_"+str(count)+ "_" + str(rcount) + "_" + str(lcount)+".sh",'w') as fw:
                    fw.write(outStr)
                    fw.close()
                lcount += 1
            rcount +=1 
        count +=1 
