#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p Lewis,BioCompute               # use the BioCompute partition Lewis,BioCompute
#SBATCH -J scvi
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores (AKA tasks)
#SBATCH --mem=128G
#################################################################
module load miniconda3
source activate conda_R
python3 -W ignore scVi_impute.py
