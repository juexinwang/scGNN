import argparse

# python generatingMethodsBatchshell_louvain.py
# python generatingMethodsBatchshell_louvain.py --imputeMode
parser = argparse.ArgumentParser(description='Generating sbatch files for HPC cluster running')
parser.add_argument('--outputDir', type=str, default='',
                    help='Directory of batch files for cluster running')
parser.add_argument('--imputeMode', action='store_true', default=False,
                    help='whether impute')
parser.add_argument('--aeOriginal', action='store_true', default=False,
                    help='whether use original')
parser.add_argument('--adjtype', type=str, default='weighted',
                    help='whether weighted')

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
    ('run_experiment_1_g_e_LK E1geK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyG1E_LK_1/'),
    ('run_experiment_1_g_f_LK E1gfK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --npyDir','npyG1F_LK_1/'),
    ('run_experiment_1_n_e_LK E1neK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useGAEembedding --npyDir','npyN1E_LK_1/'),
    # ('run_experiment_2_g_e_LK E2geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK/'),
    
    ('run_experiment_1_g_e_LK2 E1geK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding  --seed 2 --npyDir','npyG1E_LK_2/'),
    ('run_experiment_1_g_f_LK2 E1gfK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK  --seed 2 --npyDir','npyG1F_LK_2/'),
    ('run_experiment_1_n_e_LK2 E1neK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useGAEembedding --seed 2 --npyDir','npyN1E_LK_2/'),
    # ('run_experiment_2_g_e_LK2 E2geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_2/ --seed 2'),

    ('run_experiment_1_g_e_LK3 E1geK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --useGAEembedding --seed 3 --npyDir','npyG1E_LK_3/'),
    ('run_experiment_1_g_f_LK3 E1gfK','--regulized-type LTMG --EMtype EM --clustering-method LouvainK --seed 3 --npyDir','npyG1F_LK_3/'),
    ('run_experiment_1_n_e_LK3 E1neK','--regulized-type noregu --EMtype EM --clustering-method LouvainK --useGAEembedding --seed 3 --npyDir','npyN1E_LK_3/'),
    # ('run_experiment_2_g_e_LK3 E2geK','--regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding  --npyDir','npyG2E_LK_3/ --seed 3'),
    
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
    '9.Chung --n-clusters 4 --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv --aeOriginal',
    '10.Usoskin --n-clusters 11 --benchmark /home/jwang/data/scData/10.Usoskin/Usoskin_cell_label.csv',
    '11.Kolodziejczyk --n-clusters 3 --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv --aeOriginal',
    '12.Klein --n-clusters 4 --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv --aeOriginal',
    '13.Zeisel --n-clusters 7 --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv --aeOriginal'
]

paraList = [
    '--gammaPara 0.1 --regularizePara 0.0',
    '--gammaPara 0.1 --regularizePara 0.1',
    '--gammaPara 0.1 --regularizePara 0.5',
    '--gammaPara 0.1 --regularizePara 0.9',
    '--gammaPara 0.0 --regularizePara 1.0',
    ]

# generate sbatch files:
for item in methodsList:
    batchInfo,scGNNparam,outDirStr = item
    if args.aeOriginal:
        outDirStr = 'aeO/'+outDirStr
    else:
        outDirStr = 'aeC/'+outDirStr
    if args.adjtype=='weighted':
        outDirStr = 'W'+outDirStr
    elif args.adjtype=='unweighted':
        outDirStr = 'U'+outDirStr

    tmp = batchInfo.split()
    tmpstr1=tmp[0]
    tmpstr2=tmp[1]
    #difference
    # imputeStr = ''
    imputeStr = ''
    if args.imputeMode:
        tmpstr1 = tmpstr1.replace('run_experiment','run_experimentImpute')
        tmpstr2 = "I"+tmpstr2[1:]
        # tmpstr2 = "I"+tmpstr2[2:]
        imputeStr = ' --imputeMode  '
        # outDirStr = "npyImpute"+outDirStr[3:]
        # For secondary directory
        outDirStr = "npyImpute"+outDirStr[8:]
    outputFilename = args.outputDir + tmpstr1
    abbrStr = tmpstr2   

    count = 1
    for datasetName in datasetNameList:
        tcount = 0
        for para in paraList:
            commandLine = "python3 -W ignore main_benchmark_graphregu.py --datasetName "+datasetName+" "+scGNNparam+" "+outDirStr+" "+imputeStr+" "+para+"\n"
            outStr = templateStr1 + abbrStr + "_" + str(count) + templateStr2 + commandLine + "\n"
            with open(outputFilename+"_"+str(count)+"_"+str(tcount)+".sh",'w') as fw:
                fw.write(outStr)
                fw.close()
            tcount += 1
        count += 1
