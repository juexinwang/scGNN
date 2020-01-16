import argparse

parser = argparse.ArgumentParser(description='Generating sbatch files for cluster usage')
parser.add_argument('--inputFilename', type=str, default='experimentImpute_1_g_f.sh',
                    help='source file for generating sbatch files')
parser.add_argument('--inputDir', type=str, default='../',
                    help='Directory of source file for generating sbatch files')
parser.add_argument('--sbatchFilename', type=str, default='run_experimentImpute_1_g_f',
                    help='batch files for cluster running')
parser.add_argument('--outputDir', type=str, default='../',
                    help='Directory of batch files for cluster running')
parser.add_argument('--abbrStr', type=str, default='I1gf',
                    help='abbr in batch files for cluster running')
args = parser.parse_args()

inputFilename = args.inputDir + args.inputFilename
outputFilename = args.outputDir + args.sbatchFilename
abbrStr = args.abbrStr

templateStr1 = "#! /bin/bash\n"\
"######################### Batch Headers #########################\n"\
"#SBATCH -A xulab\n"\
"#SBATCH -p BioCompute               # use the BioCompute partition\n"\
"#SBATCH -J "

templateStr2 = "\n#SBATCH -o results-%j.out           # give the job output a custom name\n"\
"#SBATCH -t 2-00:00                  # two days time limit\n"\
"#SBATCH -N 1                        # number of nodes\n"\
"#SBATCH -n 8                        # number of cores (AKA tasks)\n"\
"#SBATCH --mem=64G\n"\
"#################################################################\n"\
"module load miniconda3\n"\
"source activate my_environment\n"

contList = []
with open(inputFilename) as f:
    lines = f.readlines()
    count = 1
    for line in lines:
        line = line.strip()
        if not line == '':
            contList.append(line)
            count += 1
    f.close()

count = 1
for line in contList:
    outStr = templateStr1 + abbrStr + "_" + str(count) + templateStr2 + line + '\n'
    with open(outputFilename+"_"+str(count)+".sh",'w') as fw:
        fw.write(outStr)
        fw.close()
    count += 1


