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

#Refined Matrix with --regularizePara 0.5
methodsList = [
    ('run_experiment_2_g_e E2ge','--regularizePara 0.5 --regulized-type LTMG --EMtype celltypeEM --useGAEembedding  --npyDir','npyG2E/'),
    ('run_experiment_2_g_e_Birch E2geB','--regularizePara 0.5 --regulized-type LTMG --EMtype celltypeEM --clustering-method Birch --useGAEembedding --npyDir','npyG2E_Birch/'),
    ('run_experiment_2_g_e_KMeans E2geK','--regularizePara 0.5 --regulized-type LTMG --EMtype celltypeEM --clustering-method KMeans --useGAEembedding --npyDir','npyG2E_KMeans/'),
    ('run_experiment_2_g_e_BirchN E2geN','--regularizePara 0.5 --regulized-type LTMG --EMtype celltypeEM --clustering-method BirchN --useGAEembedding --npyDir','npyG2E_BirchN/'),
    ('run_experiment_2_r_e E2re','--regularizePara 0.5 --regulized-type LTMG01 --EMtype celltypeEM --useGAEembedding  --npyDir','npyR2E/'),
    ('run_experiment_2_r_e_Birch E2reB','--regularizePara 0.5 --regulized-type LTMG01 --EMtype celltypeEM --clustering-method Birch --useGAEembedding --npyDir','npyR2E_Birch/'),
    ('run_experiment_2_r_e_KMeans E2reK','--regularizePara 0.5 --regulized-type LTMG01 --EMtype celltypeEM --clustering-method KMeans --useGAEembedding --npyDir','npyR2E_KMeans/'),
    ('run_experiment_2_r_e_BirchN E2reN','--regularizePara 0.5 --regulized-type LTMG01 --EMtype celltypeEM --clustering-method BirchN --useGAEembedding --npyDir','npyR2E_BirchN/'),
]

# select
datasetNameList = [
    '--datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --n-clusters 4',
]

reguParaList = [
    '--gammaPara 1.0 --regularizePara 0.1',
    '--gammaPara 0.5 --regularizePara 0.5',
    '--gammaPara 0.1 --regularizePara 1.0',
    '--gammaPara 0.0 --regularizePara 1.0',
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
    imputeStr = ''
    outputFilename = args.outputDir + tmpstr1
    abbrStr = tmpstr2   

    count = 1
    for datasetName in datasetNameList:
        commandLine = "python3 -W ignore main_benchmark.py --datasetName "+datasetName+" "+scGNNparam+" "+outDirStr+imputeStr+"\n"
        outStr = templateStr1 + abbrStr + "_" + str(count) + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_"+str(count)+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()
        count += 1
