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
"#SBATCH -p Lewis,BioCompute               # use the BioCompute partition\n"\
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
    ('run_experiment_1_g_b_0.8 E1gb','--regulized-type LTMG --EMtype EM --useBothembedding --npyDir','npyG1B_0.8/'),
    ('run_experiment_1_g_e_0.8 E1ge','--regulized-type LTMG --EMtype EM --useGAEembedding --npyDir','npyG1E_0.8/'),
    ('run_experiment_1_g_f_0.8 E1gf','--regulized-type LTMG --EMtype EM --npyDir','npyG1F_0.8/'),
    ('run_experiment_1_r_b_0.8 E1rb','--regulized-type LTMG01 --EMtype EM --useBothembedding --npyDir','npyR1B_0.8/'),
    ('run_experiment_1_r_e_0.8 E1re','--regulized-type LTMG01 --EMtype EM --useGAEembedding --npyDir','npyR1E_0.8/'),
    ('run_experiment_1_r_f_0.8 E1rf','--regulized-type LTMG01 --EMtype EM --npyDir','npyR1F_0.8/'),
    ('run_experiment_1_n_b_0.8 E1nb','--regulized-type noregu --EMtype EM --useBothembedding --npyDir','npyN1B_0.8/'),
    ('run_experiment_1_n_e_0.8 E1ne','--regulized-type noregu --EMtype EM --useGAEembedding --npyDir','npyN1E_0.8/'),
    ('run_experiment_1_n_f_0.8 E1nf','--regulized-type noregu --EMtype EM --npyDir','npyN1F_0.8/'),
    ('run_experiment_2_g_b_0.8 E2gb','--regulized-type LTMG --EMtype celltypeEM --useBothembedding  --npyDir','npyG2B_0.8/'),
    ('run_experiment_2_g_e_0.8 E2ge','--regulized-type LTMG --EMtype celltypeEM --useGAEembedding  --npyDir','npyG2E_0.8/'),
    ('run_experiment_2_g_f_0.8 E2gf','--regulized-type LTMG --EMtype celltypeEM --npyDir','npyG2F_0.8/'),
    ('run_experiment_2_r_b_0.8 E2rb','--regulized-type LTMG01 --EMtype celltypeEM --useBothembedding  --npyDir','npyR2B_0.8/'),
    ('run_experiment_2_r_e_0.8 E2re','--regulized-type LTMG01 --EMtype celltypeEM --useGAEembedding  --npyDir','npyR2E_0.8/'),
    ('run_experiment_2_r_f_0.8 E2rf','--regulized-type LTMG01 --EMtype celltypeEM --npyDir','npyR2F_0.8/'),
    ('run_experiment_2_n_b_0.8 E2nb','--regulized-type noregu --EMtype celltypeEM --useBothembedding --npyDir','npyN2B_0.8/'),
    ('run_experiment_2_n_e_0.8 E2ne','--regulized-type noregu --EMtype celltypeEM --useGAEembedding --npyDir','npyN2E_0.8/'),
    ('run_experiment_2_n_f_0.8 E2nf','--regulized-type noregu --EMtype celltypeEM --npyDir','npyN2F_0.8/'),

    ('run_experiment_1_g_b_LK_0.8 E1gbK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useBothembedding --npyDir','npyG1B_LK_0.8/'),
    ('run_experiment_1_g_e_LK_0.8 E1geK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyG1E_LK_0.8/'),
    ('run_experiment_1_g_f_LK_0.8 E1gfK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --npyDir','npyG1F_LK_0.8/'),
    ('run_experiment_1_r_b_LK_0.8 E1rbK','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainK --useBothembedding --npyDir','npyR1B_LK_0.8/'),
    ('run_experiment_1_r_e_LK_0.8 E1reK','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyR1E_LK_0.8/'),
    ('run_experiment_1_r_f_LK_0.8 E1rfK','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainK --npyDir','npyR1F_LK_0.8/'),
    ('run_experiment_1_n_b_LK_0.8 E1nbK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useBothembedding --npyDir','npyN1B_LK_0.8/'),
    ('run_experiment_1_n_e_LK_0.8 E1neK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyN1E_LK_0.8/'),
    ('run_experiment_1_n_f_LK_0.8 E1nfK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --npyDir','npyN1F_LK_0.8/'),
    ('run_experiment_2_g_b_LK_0.8 E2gbK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useBothembedding  --npyDir','npyG2B_LK_0.8/'),
    ('run_experiment_2_g_e_LK_0.8 E2geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_0.8/'),
    ('run_experiment_2_g_f_LK_0.8 E2gfK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyG2F_LK_0.8/'),
    ('run_experiment_2_r_b_LK_0.8 E2rbK','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --useBothembedding  --npyDir','npyR2B_LK_0.8/'),
    ('run_experiment_2_r_e_LK_0.8 E2reK','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyR2E_LK_0.8/'),
    ('run_experiment_2_r_f_LK_0.8 E2rfK','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyR2F_LK_0.8/'),
    ('run_experiment_2_n_b_LK_0.8 E2nbK','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useBothembedding --npyDir','npyN2B_LK_0.8/'),
    ('run_experiment_2_n_e_LK_0.8 E2neK','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir','npyN2E_LK_0.8/'),
    ('run_experiment_2_n_f_LK_0.8 E2nfK','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyN2F_LK_0.8/'),

    ('run_experiment_1_g_b_LB_0.8 E1gbB','--regulized-type LTMG --EMtype EM --clustering-method LouvainB --useBothembedding --npyDir','npyG1B_LB_0.8/'),
    ('run_experiment_1_g_e_LB_0.8 E1geB','--regulized-type LTMG --EMtype EM --clustering-method LouvainB --useGAEembedding --npyDir','npyG1E_LB_0.8/'),
    ('run_experiment_1_g_f_LB_0.8 E1gfB','--regulized-type LTMG --EMtype EM --clustering-method LouvainB --npyDir','npyG1F_LB_0.8/'),
    ('run_experiment_1_r_b_LB_0.8 E1rbB','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainB --useBothembedding --npyDir','npyR1B_LB_0.8/'),
    ('run_experiment_1_r_e_LB_0.8 E1reB','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainB --useGAEembedding --npyDir','npyR1E_LB_0.8/'),
    ('run_experiment_1_r_f_LB_0.8 E1rfB','--regulized-type LTMG01 --EMtype EM --clustering-method LouvainB --npyDir','npyR1F_LB_0.8/'),
    ('run_experiment_1_n_b_LB_0.8 E1nbB','--regulized-type noregu --EMtype EM --clustering-method LouvainB --useBothembedding --npyDir','npyN1B_LB_0.8/'),
    ('run_experiment_1_n_e_LB_0.8 E1neB','--regulized-type noregu --EMtype EM --clustering-method LouvainB --useGAEembedding --npyDir','npyN1E_LB_0.8/'),
    ('run_experiment_1_n_f_LB_0.8 E1nfB','--regulized-type noregu --EMtype EM --clustering-method LouvainB --npyDir','npyN1F_LB_0.8/'),
    ('run_experiment_2_g_b_LB_0.8 E2gbB','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --useBothembedding  --npyDir','npyG2B_LB_0.8/'),
    ('run_experiment_2_g_e_LB_0.8 E2geB','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','npyG2E_LB_0.8/'),
    ('run_experiment_2_g_f_LB_0.8 E2gfB','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainB --npyDir','npyG2F_LB_0.8/'),
    ('run_experiment_2_r_b_LB_0.8 E2rbB','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --useBothembedding  --npyDir','npyR2B_LB_0.8/'),
    ('run_experiment_2_r_e_LB_0.8 E2reB','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding  --npyDir','npyR2E_LB_0.8/'),
    ('run_experiment_2_r_f_LB_0.8 E2rfB','--regulized-type LTMG01 --EMtype celltypeEM --clustering-method LouvainB --npyDir','npyR2F_LB_0.8/'),
    ('run_experiment_2_n_b_LB_0.8 E2nbB','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainB --useBothembedding --npyDir','npyN2B_LB_0.8/'),
    ('run_experiment_2_n_e_LB_0.8 E2neB','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainB --useGAEembedding --npyDir','npyN2E_LB_0.8/'),
    ('run_experiment_2_n_f_LB_0.8 E2nfB','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainB --npyDir','npyN2F_LB_0.8/'),
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
    '1.Biase --n-clusters 3 --benchmark /home/jwang/data/scData/1.Biase/Biase_cell_label.csv --resolution 0.8',
    '2.Li --n-clusters 9 --benchmark /home/jwang/data/scData/2.Li/Li_cell_label.csv --resolution 0.8',
    '3.Treutlein --n-clusters 5 --benchmark /home/jwang/data/scData/3.Treutlein/Treutlein_cell_label.csv --resolution 0.8',
    '4.Yan --n-clusters 7 --benchmark /home/jwang/data/scData/4.Yan/Yan_cell_label.csv --resolution 0.8',
    '5.Goolam --n-clusters 5 --benchmark /home/jwang/data/scData/5.Goolam/Goolam_cell_label.csv --resolution 0.8',
    '6.Guo --n-clusters 9 --benchmark /home/jwang/data/scData/6.Guo/Guo_cell_label.csv --resolution 0.8',
    '7.Deng --n-clusters 10 --benchmark /home/jwang/data/scData/7.Deng/Deng_cell_label.csv --resolution 0.8',
    '8.Pollen --n-clusters 11 --benchmark /home/jwang/data/scData/8.Pollen/Pollen_cell_label.csv --resolution 0.8',
    '9.Chung --n-clusters 4 --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv --resolution 0.8',
    '10.Usoskin --n-clusters 11 --benchmark /home/jwang/data/scData/10.Usoskin/Usoskin_cell_label.csv --resolution 0.8',
    '11.Kolodziejczyk --n-clusters 3 --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv --resolution 0.8',
    '12.Klein --n-clusters 4 --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv --resolution 0.8',
    '13.Zeisel --n-clusters 7 --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv --resolution 0.8'
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
