inputfilename = 'experimentImpute_1_g_f.sh'
outputfilename = 'run_experimentImpute_1_g'
abbrStr = 'I1gf'

templateStr1 = "#! /bin/bash\n"\
"######################### Batch Headers #########################\n"\
"#SBATCH -A xulab\n"\
"#SBATCH -p BioCompute               # use the BioCompute partition\n"\
"#SBATCH -J "

templateStr2 = "\n#SBATCH -o results-%j.out           # give the job output a custom name\n"\
"#SBATCH -t 2-00:00                  # two days time limit\n"\
"#SBATCH -N 1                        # number of nodes\n"\
"#SBATCH -n 4                        # number of cores (AKA tasks)\n"\
"#SBATCH --mem=32G\n"\
"#################################################################\n"\
"module load miniconda3\n"\
"source activate my_environment\n"

contList = []
with open(inputfilename) as f:
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
    outStr = templateStr1 + abbrStr + "_" + str(count) + templateStr2 + line
    with open(outputfilename+"_"+str(count)+".sh",'w') as fw:
        fw.write(outStr)
        fw.close()
    count += 1


