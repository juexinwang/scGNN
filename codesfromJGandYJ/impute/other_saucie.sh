#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p Lewis,BioCompute               # use the BioCompute partition Lewis,BioCompute
#SBATCH -J saucie
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores (AKA tasks)
#SBATCH --mem=128G
#################################################################

module load miniconda3
source activate /storage/htc/joshilab/wangjue/conda_R_saucie
# source activate /storage/htc/joshilab/wangjue/conda_R_gpu
# module load cuda/cuda-10.1.243
python3 -W ignore SAUCIE_impute.py