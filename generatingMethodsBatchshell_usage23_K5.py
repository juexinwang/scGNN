import argparse

# python generatingMethodsBatchshell_louvain.py
# python generatingMethodsBatchshell_louvain.py --imputeMode
parser = argparse.ArgumentParser(description='Generating sbatch files for HPC cluster running')
parser.add_argument('--outputDir', type=str, default='',
                    help='Directory of batch files for cluster running')
parser.add_argument('--imputeMode', action='store_true', default=True,
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
    ('run_experiment_1_g_e_LK E1geK','--k 5 --regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyG1E_LK5/'),
    ('run_experiment_2_g_e_LK E2geK','--k 5 --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK5/'),
    ('run_experiment_2_g_f_LK E2gfK','--k 5 --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyG2F_LK5/'),
    ('run_experiment_2_n_e_LK E2neK','--k 5 --regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir','npyN2E_LK5/'),

    ('run_experiment_1_g_e_LK_2 21geK','--k 5 --regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyG1E_LK5_2/ --seed 2'),
    ('run_experiment_2_g_e_LK_2 22geK','--k 5 --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK5_2/ --seed 2'),
    ('run_experiment_2_g_f_LK_2 22gfK','--k 5 --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyG2F_LK5_2/ --seed 2'),
    ('run_experiment_2_n_e_LK_2 22neK','--k 5 --regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir','npyN2E_LK5_2/ --seed 2'),

    ('run_experiment_1_g_e_LK_3 31geK','--k 5 --regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyG1E_LK5_3/ --seed 3'),
    ('run_experiment_2_g_e_LK_3 32geK','--k 5 --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK5_3/ --seed 3'),
    ('run_experiment_2_g_f_LK_3 32gfK','--k 5 --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --npyDir','npyG2F_LK5_3/  --seed 3'),
    ('run_experiment_2_n_e_LK_3 32neK','--k 5 --regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir','npyN2E_LK5_3/  --seed 3'),

]

# select
datasetNameList = [
    '1.Biase --n-clusters 3 --benchmark /home/jwang/data/scData/1.Biase/Biase_cell_label.csv',
    '2.Li --n-clusters 9 --benchmark /home/jwang/data/scData/2.Li/Li_cell_label.csv',
    '3.Treutlein --n-clusters 5 --benchmark /home/jwang/data/scData/3.Treutlein/Treutlein_cell_label.csv',
    '4.Yan --n-clusters 7 --benchmark /home/jwang/data/scData/4.Yan/Yan_cell_label.csv',
    '5.Goolam --n-clusters 5 --benchmark /home/jwang/data/scData/5.Goolam/Goolam_cell_label.csv',
    '6.Guo --n-clusters 9 --benchmark /home/jwang/data/scData/6.Guo/Guo_cell_label.csv',
    '7.Deng --n-clusters 10 --benchmark /home/jwang/data/scData/7.Deng/Deng_cell_label.csv',
    '8.Pollen --n-clusters 11 --benchmark /home/jwang/data/scData/8.Pollen/Pollen_cell_label.csv',
    '9.Chung --n-clusters 4 --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv',
    '10.Usoskin --n-clusters 11 --benchmark /home/jwang/data/scData/10.Usoskin/Usoskin_cell_label.csv',
    '11.Kolodziejczyk --n-clusters 3 --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv',
    '12.Klein --n-clusters 4 --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv',
    '13.Zeisel --n-clusters 7 --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv'
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
