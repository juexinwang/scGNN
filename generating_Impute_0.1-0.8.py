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
"#SBATCH -p Lewis,BioCompute               # use the BioCompute partition Lewis,BioCompute\n"\
"#SBATCH -J "

templateStr2 = "\n#SBATCH -o results-%j.out           # give the job output a custom name\n"\
"#SBATCH -t 2-00:00                  # two days time limit\n"\
"#SBATCH -N 1                        # number of nodes\n"\
"#SBATCH -n 1                        # number of cores (AKA tasks)\n"\
"#SBATCH --mem=128G\n"\
"#################################################################\n"\
"module load miniconda3\n"\
"source activate conda_R\n"

#tuple list
#batchInfo,scGNNparam,outDir
#huge matrix
methodsList = [
    ('run_experiment_2_g_e_LK_1 E2geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_1a/ --seed 1'),
    ('run_experiment_2_g_e_LK_2 22geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_2a/ --seed 2'),
    ('run_experiment_2_g_e_LK_3 32geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_3a/ --seed 3'),
    ('run_experiment_2_g_e_LK_4 42geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_4a/ --seed 4'),
    ('run_experiment_2_g_e_LK_5 52geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_5a/ --seed 5'),
    ('run_experiment_2_g_e_LK_6 62geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_6a/ --seed 6'),
    ('run_experiment_2_g_e_LK_7 72geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_7a/ --seed 7'),
    ('run_experiment_2_g_e_LK_8 82geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_8a/ --seed 8'),
    ('run_experiment_2_g_e_LK_9 92geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_9a/ --seed 9'),
    ('run_experiment_2_g_e_LK_10 02geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_10a/ --seed 10'),
]

dropoutList = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8']

# generate sbatch files:
for item in methodsList:
    batchInfo,scGNNparam,outDirStr = item
    tmp = batchInfo.split()
    tmpstr1=tmp[0]
    tmpstr2=tmp[1]
    imputeStr = ''
    if args.imputeMode:
        tmpstr1 = tmpstr1.replace('run_experiment','run_experimentImpute')
        tmpstr2 = "I"+tmpstr2
        # tmpstr2 = "I"+tmpstr2[2:]
        imputeStr = ' --imputeMode  '
        outDirStr = "npyImpute"+outDirStr[3:]
    outputFilename = args.outputDir + tmpstr1
    abbrStr = tmpstr2   

    for dropoutPara in dropoutList:
        commandLine = "python3 -W ignore main_benchmark.py --datasetName 12.Klein --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv "+scGNNparam+" "+outDirStr+imputeStr+" --dropoutRatio "+dropoutPara+"\n"
        outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_12_"+dropoutPara+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()
