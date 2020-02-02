import argparse

# python generatingMethodsBatchshell.py
# python generatingMethodsBatchshell.py --imputeMode
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
"#SBATCH -n 8                        # number of cores (AKA tasks)\n"\
"#SBATCH --mem=128G\n"\
"#################################################################\n"\
"module load miniconda3\n"\
"source activate conda_R\n"

#tuple list
#batchInfo,scGNNparam,outDir
methodsList = [
    ('run_experiment_1_g_e E1ge','--EMtype EM --useGAEembedding --npyDir','npyG1E/'),
    ('run_experiment_1_g_f E1gf','--EMtype EM --npyDir','npyG1F/'),
    ('run_experiment_1_n_e E1ne','--regulized-type noregu --EMtype EM --useGAEembedding --npyDir','npyN1E/'),
    ('run_experiment_1_n_f E1nf','--regulized-type noregu --EMtype EM --npyDir','npyN1F/'),
    ('run_experiment_2_g_e E2ge','--EMtype celltypeEM --useGAEembedding  --npyDir','npyG2E/'),
    ('run_experiment_2_g_f E2gf','--EMtype celltypeEM --npyDir','npyG2F/'),
    ('run_experiment_2_n_e E2ne','--regulized-type noregu --EMtype celltypeEM --useGAEembedding --npyDir','npyN2E/'),
    ('run_experiment_2_n_f E2nf','--regulized-type noregu --EMtype celltypeEM --npyDir','npyN2F/'),
    ('run_experiment_2_g_e_AffinityPropagation E2geA','--EMtype celltypeEM --clustering-method AffinityPropagation --useGAEembedding --npyDir','npyG2E_AffinityPropagation/'),
    ('run_experiment_2_g_e_AgglomerativeClustering E2geG','--EMtype celltypeEM --clustering-method AgglomerativeClustering --useGAEembedding --npyDir','npyG2E_AgglomerativeClustering/'),
    ('run_experiment_2_g_e_Birch E2geB','--EMtype celltypeEM --clustering-method Birch --useGAEembedding --npyDir','npyG2E_Birch/'),
    ('run_experiment_2_g_e_KMeans E2geK','--EMtype celltypeEM --clustering-method KMeans --useGAEembedding --npyDir','npyG2E_KMeans/'),
    ('run_experiment_2_g_e_SpectralClustering E2geS','--EMtype celltypeEM --clustering-method SpectralClustering --useGAEembedding --npyDir','npyG2E_SpectralClustering/'),
    ('run_experiment_2_g_f_AffinityPropagation E2gfA','--EMtype celltypeEM --clustering-method AffinityPropagation --npyDir','npyG2F_AffinityPropagation/'),
    ('run_experiment_2_g_f_AgglomerativeClustering E2gfG','--EMtype celltypeEM --clustering-method AgglomerativeClustering --npyDir','npyG2F_AgglomerativeClustering/'),
    ('run_experiment_2_g_f_Birch E2gfB','--EMtype celltypeEM --clustering-method Birch --npyDir','npyG2F_Birch/'),
    ('run_experiment_2_g_f_KMeans E2gfK','--EMtype celltypeEM --clustering-method KMeans --npyDir','npyG2F_KMeans/'),
    ('run_experiment_2_g_f_SpectralClustering E2gfS','--EMtype celltypeEM --clustering-method SpectralClustering --npyDir','npyG2F_SpectralClustering/'),
    ('run_experiment_2_n_e_AffinityPropagation E2neA','--regulized-type noregu --EMtype celltypeEM --clustering-method AffinityPropagation --useGAEembedding --npyDir','npyN2E_AffinityPropagation/'),
    ('run_experiment_2_n_e_AgglomerativeClustering E2neG','--regulized-type noregu --EMtype celltypeEM --clustering-method AgglomerativeClustering --useGAEembedding --npyDir','npyN2E_AgglomerativeClustering/'),
    ('run_experiment_2_n_e_Birch E2neB','--regulized-type noregu --EMtype celltypeEM --clustering-method Birch --useGAEembedding --npyDir','npyN2E_Birch/'),
    ('run_experiment_2_n_e_KMeans E2neK','--regulized-type noregu --EMtype celltypeEM --clustering-method KMeans --useGAEembedding --npyDir','npyN2E_KMeans/'),
    ('run_experiment_2_n_e_SpectralClustering E2neS','--regulized-type noregu --EMtype celltypeEM --clustering-method SpectralClustering --useGAEembedding --npyDir','npyN2E_SpectralClustering/'),
    ('run_experiment_2_n_f_AffinityPropagation E2nfA','--regulized-type noregu --EMtype celltypeEM --clustering-method AffinityPropagation --npyDir','npyN2F_AffinityPropagation/'),
    ('run_experiment_2_n_f_AgglomerativeClustering E2nfG','--regulized-type noregu --EMtype celltypeEM --clustering-method AgglomerativeClustering --npyDir','npyN2F_AgglomerativeClustering/'),
    ('run_experiment_2_n_f_Birch E2nfB','--regulized-type noregu --EMtype celltypeEM --clustering-method Birch --npyDir','npyN2F_Birch/'),
    ('run_experiment_2_n_f_KMeans E2nfK','--regulized-type noregu --EMtype celltypeEM --clustering-method KMeans --npyDir','npyN2F_KMeans/'),
    ('run_experiment_2_n_f_SpectralClustering E2nfS','--regulized-type noregu --EMtype celltypeEM --clustering-method SpectralClustering --npyDir','npyN2F_SpectralClustering/')
]

datasetNameList = [
    'MMPbasal_2000',
    'MMPbasal_2000 --discreteTag',
    'MMPbasal_2000_LTMG',
    '4.Yan --n-clusters 7',
    '4.Yan --discreteTag --n-clusters 7',
    '4.Yan_LTMG --n-clusters 7',
    '5.Goolam --n-clusters 5',
    '5.Goolam --discreteTag --n-clusters 5',
    '5.Goolam_LTMG --n-clusters 5',
    '7.Deng --n-clusters 10',
    '7.Deng --discreteTag --n-clusters 10',
    '7.Deng_LTMG --n-clusters 10'
    '8.Pollen --n-clusters 11',
    '8.Pollen --discreteTag --n-clusters 11',
    '8.Pollen_LTMG --n-clusters 11',
    '11.Kolodziejczyk --n-clusters 3',
    '11.Kolodziejczyk --discreteTag --n-clusters 3',
    '11.Kolodziejczyk_LTMG --n-clusters 3'
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
        imputeStr = ' --imputeMode  '
        outDirStr = "npyImpute"+outDirStr[3:]
    outputFilename = args.outputDir + tmpstr1
    abbrStr = tmpstr2   

    count = 1
    for datasetName in datasetNameList:
        commandLine = "python3 -W ignore main.py --datasetName "+datasetName+" "+scGNNparam+" "+outDirStr+imputeStr+"\n"
        outStr = templateStr1 + abbrStr + "_" + str(count) + templateStr2 + commandLine + "\n"
        with open(outputFilename+"_"+str(count)+".sh",'w') as fw:
            fw.write(outStr)
            fw.close()
        count += 1
