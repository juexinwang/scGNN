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
    ('run_case_2geLK E1','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','casenpyG2E_LK/'),
    ('run_case_2reLK E2','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','casenpyR2E_LK/'),
    ('run_case_2geLB E3','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','casenpyG2E_LB/'),
    ('run_case_2reLB E4','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','casenpyR2E_LB/'),    
]

# select
datasetNameList = [
    'AD_GSE138852_2x8CT',
    'AD_GSE103334_NORMED_8CT',
    'E-GEOD-139324',
    'liver_cancer_GSE98638_11CT',
    'liver_cancer_10X_20CT',
    'liver_cancer_smart_22CT',
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
