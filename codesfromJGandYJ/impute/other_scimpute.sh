#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p BioCompute,Lewis               # use the BioCompute partition Lewis,BioCompute
#SBATCH -J scimpute
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 12                        # number of cores (AKA tasks)
#SBATCH --mem=128G
#################################################################
module load miniconda3
source activate conda_R
# python3 -W ignore SCIMPUTE_impute.py
python3 -W ignore SCIMPUTE_impute.py --origin