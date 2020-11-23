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
"#SBATCH -p BioCompute,Lewis               # use the BioCompute partition Lewis,BioCompute\n"\
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
    ('run_experiment_2_g_e_L_1 2geL1','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --L1Para 0.0 --seed 1 --npyDir','npyG2EL_1/'),
    ('run_experiment_1_g_e_1 1ge1','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --seed 1 --npyDir','npyG1E_1/'),
    ('run_experiment_2_g_f_1 2gf1','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --seed 1 --npyDir','npyG2F_1/'),
    ('run_experiment_2_n_e_1 2ne1','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --seed 1 --npyDir','npyN2E_1/'),
    ('run_experiment_2_g_e_1 2ge1','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --seed 1 --npyDir','npyG2E_1/'),

    ('run_experiment_2_g_e_L_2 2geL2','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --L1Para 0.0 --seed 2 --npyDir','npyG2EL_2/'),
    ('run_experiment_1_g_e_2 1ge2','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --seed 2 --npyDir','npyG1E_2/'),
    ('run_experiment_2_g_f_2 2gf2','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --seed 2 --npyDir','npyG2F_2/'),
    ('run_experiment_2_n_e_2 2ne2','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --seed 2 --npyDir','npyN2E_2/'),
    ('run_experiment_2_g_e_2 2ge2','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --seed 2 --npyDir','npyG2E_2/'),

    ('run_experiment_2_g_e_L_3 2geL3','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --L1Para 0.0 --seed 3 --npyDir','npyG2EL_3/'),
    ('run_experiment_1_g_e_3 1ge3','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --seed 3 --npyDir','npyG1E_3/'),
    ('run_experiment_2_g_f_3 2gf3','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --seed 3 --npyDir','npyG2F_3/'),
    ('run_experiment_2_n_e_3 2ne3','--regulized-type noregu --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --seed 3 --npyDir','npyN2E_3/'),
    ('run_experiment_2_g_e_3 2ge3','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --seed 3 --npyDir','npyG2E_3/'),
]

dropoutList = ['0.1','0.3','0.6','0.8']

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
        commandLine = "python3 -W ignore main_benchmark.py --datasetName 9.Chung --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv "+scGNNparam+" "+outDirStr+imputeStr+" --dropoutRatio "+dropoutPara+"\n"
        outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_9_"+dropoutPara+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()

    for dropoutPara in dropoutList:
        commandLine = "python3 -W ignore main_benchmark.py --datasetName 11.Kolodziejczyk --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv "+scGNNparam+" "+outDirStr+imputeStr+" --dropoutRatio "+dropoutPara+"\n"
        outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_11_"+dropoutPara+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()  

    for dropoutPara in dropoutList:
        commandLine = "python3 -W ignore main_benchmark.py --datasetName 12.Klein --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv "+scGNNparam+" "+outDirStr+imputeStr+" --dropoutRatio "+dropoutPara+"\n"
        outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_12_"+dropoutPara+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()

    for dropoutPara in dropoutList:
        commandLine = "python3 -W ignore main_benchmark.py --datasetName 13.Zeisel --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv "+scGNNparam+" "+outDirStr+imputeStr+" --dropoutRatio "+dropoutPara+"\n"
        outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_13_"+dropoutPara+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()
