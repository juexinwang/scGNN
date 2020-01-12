#! /bin/bash
######################### Batch Headers #########################
#SBATCH -p Lewis                    # use the Lewis partition
#SBATCH -J eI2geSC_9                  # give the job a custom name
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two hour time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 4                        # number of cores (AKA tasks)
#SBATCH --mem=32G
#################################################################

module load miniconda3
source activate my_environment

# Now you can run Python scripts that use the packages in your environment
python3 -W ignore main.py --datasetName MMPbasal_2000 --EMtype celltypeEM --clustering-method SpectralClustering --useGAEembedding --imputeMode --npyDir npyImputeG2E_SpectralClustering/
