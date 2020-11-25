#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH -A xulab
#SBATCH --partition gpu4
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=12G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes
#SBATCH -J SAUCIE
#SBATCH --gres gpu:1 #gpu:1 any gpu
## labels and outputs
#SBATCH --job-name=modelpyenetCB-%j.out
#SBATCH --output=results-%j.out  # %j is the unique jobID
#################################################################

module load miniconda3
source activate /storage/htc/joshilab/wangjue/conda_R_gpu
module load cuda/cuda-10.1.243
python3 -W ignore SAUCIE_impute.py