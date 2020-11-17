import argparse

# python generatingMethodsBatchshell_louvain.py
# python generatingMethodsBatchshell_louvain.py --imputeMode
parser = argparse.ArgumentParser(description='Generating sbatch files for HPC cluster running')
parser.add_argument('--outputDir', type=str, default='',
                    help='Directory of batch files for cluster running')
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
    ('plot_G2E_0.1 G2E1','LTMG_0.1_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),
    ('plot_G2E_0.3 G2E3','LTMG_0.3_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),
    ('plot_G2E_0.6 G2E6','LTMG_0.6_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),
    ('plot_G2E_0.8 G2E8','LTMG_0.8_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),

    ('plot_G2EL_0.1 G2E1','LTMG_0.1_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),
    ('plot_G2EL_0.3 G2E3','LTMG_0.3_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),
    ('plot_G2EL_0.6 G2E6','LTMG_0.6_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),
    ('plot_G2EL_0.8 G2E8','LTMG_0.8_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2E'),

    ('plot_G1E_0.1 G1E1','LTMG_0.1_10-0.1-0.9-0.0-0.3-0.1','npyImputeG1E'),
    ('plot_G1E_0.3 G1E3','LTMG_0.3_10-0.1-0.9-0.0-0.3-0.1','npyImputeG1E'),
    ('plot_G1E_0.6 G1E6','LTMG_0.6_10-0.1-0.9-0.0-0.3-0.1','npyImputeG1E'),
    ('plot_G1E_0.8 G1E8','LTMG_0.8_10-0.1-0.9-0.0-0.3-0.1','npyImputeG1E'),

    ('plot_G2F_0.1 G2F1','LTMG_0.1_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2F'),
    ('plot_G2F_0.3 G2F3','LTMG_0.3_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2F'),
    ('plot_G2F_0.6 G2F6','LTMG_0.6_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2F'),
    ('plot_G2F_0.8 G2F8','LTMG_0.8_10-0.1-0.9-0.0-0.3-0.1','npyImputeG2F'),

    ('plot_N2E_0.1 N2E1','noregu_0.1_10-0.1-0.9-0.0-0.3-0.1','npyImputeN2E'),
    ('plot_N2E_0.3 N2E3','noregu_0.3_10-0.1-0.9-0.0-0.3-0.1','npyImputeN2E'),
    ('plot_N2E_0.6 N2E6','noregu_0.6_10-0.1-0.9-0.0-0.3-0.1','npyImputeN2E'),
    ('plot_N2E_0.8 N2E8','noregu_0.8_10-0.1-0.9-0.0-0.3-0.1','npyImputeN2E'),

]

seedList = ['_1/','_2/','_3/']

# generate sbatch files:
for item in methodsList:
    batchInfo,param,dirStr = item
    tmp = batchInfo.split()
    tmpstr1=tmp[0]
    tmpstr2=tmp[1]
    imputeStr = ''
    outputFilename = args.outputDir + tmpstr1
    abbrStr = tmpstr2   

    commandLine = ''
    for seed in seedList:
        commandLine += "python3 -W ignore main_benchmark.py --datasetName 12.Klein --para "+param+" --inDir "+dirStr+seed+" --outDir "+dirStr+seed+"\n"
        commandLine += "R CMD BATCH plot_distribution.r 12.Klein "+param+" "+dirStr+seed+" "+dirStr+seed+"\n"
    outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
    with open(outputFilename+"_12.sh",'w') as fw:
        fw.write(outStr)
        fw.close()

    commandLine = ''
    for seed in seedList:
        commandLine += "python3 -W ignore main_benchmark.py --datasetName 13.Zeisel --para "+param+" --inDir "+dirStr+seed+" --outDir "+dirStr+seed+"\n"
        commandLine += "R CMD BATCH plot_distribution.r 13.Zeisel "+param+" "+dirStr+seed+" "+dirStr+seed+"\n"
    outStr = templateStr1 + abbrStr + templateStr2 + commandLine + "\n"
    with open(outputFilename+"_13.sh",'w') as fw:
        fw.write(outStr)
        fw.close()

