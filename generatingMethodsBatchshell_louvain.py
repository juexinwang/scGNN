import argparse

# python generatingMethodsBatchshell_louvain.py
# python generatingMethodsBatchshell_louvain.py --imputeMode
parser = argparse.ArgumentParser(description='Generating sbatch files for HPC cluster running')
parser.add_argument('--outputDir', type=str, default='',
                    help='Directory of batch files for cluster running')
parser.add_argument('--imputeMode', action='store_true', default=False,
                    help='whether impute')
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

#tuple list
#batchInfo,scGNNparam,outDir
#huge matrix
methodsList = [
    ('run_experiment_1_g_b E1gb','--regulized-type LTMG --EMtype EM --useBothembedding --npyDir','npyG1B/'),
    ('run_experiment_1_g_e E1ge','--regulized-type LTMG --EMtype EM --useGAEembedding --npyDir','npyG1E/'),
    ('run_experiment_1_g_f E1gf','--regulized-type LTMG --EMtype EM --npyDir','npyG1F/'),
    ('run_experiment_1_r_b E1rb','--regulized-type LTMG01 --EMtype EM --useBothembedding --npyDir','npyR1B/'),
    ('run_experiment_1_r_e E1re','--regulized-type LTMG01 --EMtype EM --useGAEembedding --npyDir','npyR1E/'),
    ('run_experiment_1_r_f E1rf','--regulized-type LTMG01 --EMtype EM --npyDir','npyR1F/'),
    ('run_experiment_1_n_b E1nb','--regulized-type noregu --EMtype EM --useBothembedding --npyDir','npyN1B/'),
    ('run_experiment_1_n_e E1ne','--regulized-type noregu --EMtype EM --useGAEembedding --npyDir','npyN1E/'),
    ('run_experiment_1_n_f E1nf','--regulized-type noregu --EMtype EM --npyDir','npyN1F/'),
    ('run_experiment_2_g_b E2gb','--regulized-type LTMG --EMtype celltypeEM --useBothembedding  --npyDir','npyG2B/'),
    ('run_experiment_2_g_e E2ge','--regulized-type LTMG --EMtype celltypeEM --useGAEembedding  --npyDir','npyG2E/'),
    ('run_experiment_2_g_f E2gf','--regulized-type LTMG --EMtype celltypeEM --npyDir','npyG2F/'),
    ('run_experiment_2_r_b E2rb','--regulized-type LTMG01 --EMtype celltypeEM --useBothembedding  --npyDir','npyR2B/'),
    ('run_experiment_2_r_e E2re','--regulized-type LTMG01 --EMtype celltypeEM --useGAEembedding  --npyDir','npyR2E/'),
    ('run_experiment_2_r_f E2rf','--regulized-type LTMG01 --EMtype celltypeEM --npyDir','npyR2F/'),
    ('run_experiment_2_n_b E2nb','--regulized-type noregu --EMtype celltypeEM --useBothembedding --npyDir','npyN2B/'),
    ('run_experiment_2_n_e E2ne','--regulized-type noregu --EMtype celltypeEM --useGAEembedding --npyDir','npyN2E/'),
    ('run_experiment_2_n_f E2nf','--regulized-type noregu --EMtype celltypeEM --npyDir','npyN2F/'),

    ('run_experiment_1_g_b_LK E1gbK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useBothembedding --npyDir','npyG1B_LK/'),
    ('run_experiment_1_g_e_LK E1geK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyG1E_LK/'),
    ('run_experiment_1_g_f_LK E1gfK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --npyDir','npyG1F_LK/'),
    ('run_experiment_1_r_b_LK E1rbK','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainK --useBothembedding --npyDir','npyR1B_LK/'),
    ('run_experiment_1_r_e_LK E1reK','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyR1E_LK/'),
    ('run_experiment_1_r_f_LK E1rfK','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainK --npyDir','npyR1F_LK/'),
    ('run_experiment_1_n_b_LK E1nbK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useBothembedding --npyDir','npyN1B_LK/'),
    ('run_experiment_1_n_e_LK E1neK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyN1E_LK/'),
    ('run_experiment_1_n_f_LK E1nfK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --npyDir','npyN1F_LK/'),
    ('run_experiment_2_g_b_LK E2gbK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useBothembedding  --npyDir','npyG2B_LK/'),
    ('run_experiment_2_g_e_LK E2geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_2/'),
    ('run_experiment_2_g_f_LK E2gfK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyG2F_LK/'),
    ('run_experiment_2_r_b_LK E2rbK','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --useBothembedding  --npyDir','npyR2B_LK/'),
    ('run_experiment_2_r_e_LK E2reK','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyR2E_LK_2/'),
    ('run_experiment_2_r_f_LK E2rfK','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyR2F_LK/'),
    ('run_experiment_2_n_b_LK E2nbK','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useBothembedding --npyDir','npyN2B_LK/'),
    ('run_experiment_2_n_e_LK E2neK','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir','npyN2E_LK/'),
    ('run_experiment_2_n_f_LK E2nfK','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyN2F_LK/'),

    ('run_experiment_1_g_b_LB E1gbB','--regulized-type LTMG --EMtype EM --clustering-method LouvainB --useBothembedding --npyDir','npyG1B_LB/'),
    ('run_experiment_1_g_e_LB E1geB','--regulized-type LTMG --EMtype EM --clustering-method LouvainB --useGAEembedding --npyDir','npyG1E_LB/'),
    ('run_experiment_1_g_f_LB E1gfB','--regulized-type LTMG --EMtype EM --clustering-method LouvainB --npyDir','npyG1F_LB/'),
    ('run_experiment_1_r_b_LB E1rbB','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainB --useBothembedding --npyDir','npyR1B_LB/'),
    ('run_experiment_1_r_e_LB E1reB','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainB --useGAEembedding --npyDir','npyR1E_LB/'),
    ('run_experiment_1_r_f_LB E1rfB','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainB --npyDir','npyR1F_LB/'),
    ('run_experiment_1_n_b_LB E1nbB','--regulized-type noregu --EMtype EM --clustering-method LouvainB --useBothembedding --npyDir','npyN1B_LB/'),
    ('run_experiment_1_n_e_LB E1neB','--regulized-type noregu --EMtype EM --clustering-method LouvainB --useGAEembedding --npyDir','npyN1E_LB/'),
    ('run_experiment_1_n_f_LB E1nfB','--regulized-type noregu --EMtype EM --clustering-method LouvainB --npyDir','npyN1F_LB/'),
    ('run_experiment_2_g_b_LB E2gbB','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --useBothembedding  --npyDir','npyG2B_LB/'),
    ('run_experiment_2_g_e_LB E2geB','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','npyG2E_LB_2/'),
    ('run_experiment_2_g_f_LB E2gfB','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --npyDir','npyG2F_LB/'),
    ('run_experiment_2_r_b_LB E2rbB','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --useBothembedding  --npyDir','npyR2B_LB/'),
    ('run_experiment_2_r_e_LB E2reB','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','npyR2E_LB_2/'),
    ('run_experiment_2_r_f_LB E2rfB','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --npyDir','npyR2F_LB/'),
    ('run_experiment_2_n_b_LB E2nbB','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainB --useBothembedding --npyDir','npyN2B_LB/'),
    ('run_experiment_2_n_e_LB E2neB','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding --npyDir','npyN2E_LB/'),
    ('run_experiment_2_n_f_LB E2nfB','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainB --npyDir','npyN2F_LB/'),
]



# All
# datasetNameList = [
#     '1.Biase --n-clusters 3',
#     '1.Biase --discreteTag --n-clusters 3',
#     '2.Li --n-clusters 9',
#     '2.Li --discreteTag --n-clusters 9',
#     '3.Treutlein --n-clusters 5',
#     '3.Treutlein --discreteTag --n-clusters 5',
#     '4.Yan --n-clusters 7',
#     '4.Yan --discreteTag --n-clusters 7',
#     '5.Goolam --n-clusters 5',
#     '5.Goolam --discreteTag --n-clusters 5',
#     '6.Guo --n-clusters 9',
#     '6.Guo --discreteTag --n-clusters 9',
#     '7.Deng --n-clusters 10',
#     '7.Deng --discreteTag --n-clusters 10',
#     '8.Pollen --n-clusters 11',
#     '8.Pollen --discreteTag --n-clusters 11',
#     '9.Chung --n-clusters 4',
#     '9.Chung --discreteTag --n-clusters 4',
#     '10.Usoskin --n-clusters 11',
#     '10.Usoskin --discreteTag --n-clusters 11',
#     '11.Kolodziejczyk --n-clusters 3',
#     '11.Kolodziejczyk --discreteTag --n-clusters 3',
#     '12.Klein --n-clusters 4',
#     '12.Klein --discreteTag --n-clusters 4',
#     '13.Zeisel --n-clusters 7',
#     '13.Zeisel --discreteTag --n-clusters 7',
#     '20.10X_2700_seurat',
#     '20.10X_2700_seurat --discreteTag',
#     '30.Schafer',
#     '30.Schafer --discreteTag'
# ]

# select
datasetNameList = [
    '1.Biase --n-clusters 3',
    '2.Li --n-clusters 9',
    '3.Treutlein --n-clusters 5',
    '4.Yan --n-clusters 7',
    '5.Goolam --n-clusters 5',
    '6.Guo --n-clusters 9',
    '7.Deng --n-clusters 10',
    '8.Pollen --n-clusters 11',
    '9.Chung --n-clusters 4',
    '10.Usoskin --n-clusters 11',
    '11.Kolodziejczyk --n-clusters 3',
    '12.Klein --n-clusters 4',
    '13.Zeisel --n-clusters 7'
]

# generate sbatch files:
for item in methodsList:
    batchInfo,scGNNparam,outDirStr = item
    tmp = batchInfo.split()
    tmpstr1=tmp[0]
    tmpstr2=tmp[1]
    imputeStr = ''
    if args.imputeMode:
        tmpstr1 = tmpstr1.replace('run_experiment','run_experimentImpute')
        tmpstr2 = "I"+tmpstr2[1:]
        # tmpstr2 = "I"+tmpstr2[2:]
        imputeStr = ' --imputeMode  '
        outDirStr = "npyImpute"+outDirStr[3:]
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
