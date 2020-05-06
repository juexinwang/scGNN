#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p Lewis,BioCompute               # use the BioCompute partition
#SBATCH -J 13R1
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 2                        # number of cores (AKA tasks)
#SBATCH --mem=128G
#################################################################
module load miniconda3
source activate conda_R
python3 -W ignore main_benchmark_graphregu.py --datasetName 13.Zeisel --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv --EMtype EM --EMregulized-type GraphR --ONEregulized-type LTMG-GraphR --useGAEembedding --npyDir npyeRG1E/ --imputeMode  --EM-epochs 5