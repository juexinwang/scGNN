#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p Lewis,BioCompute               # use the BioCompute partition Lewis,BioCompute
#SBATCH -J L_magic
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores (AKA tasks)
#SBATCH --mem=128G
#################################################################
module load miniconda3
source activate conda_R
python -W ignore louvain.py --input othermethods/magic/9.Chung_0.0_1_recon.npy --output otherresults/magic/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/magic/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/magic/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/magic/12.Klein_0.0_1_recon.npy --output otherresults/magic/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/magic/13.Zeisel_0.0_1_recon.npy --output otherresults/magic/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/dca/9.Chung_0.0_1_recon.npy --output otherresults/dca/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/dca/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/dca/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/dca/12.Klein_0.0_1_recon.npy --output otherresults/dca/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/dca/13.Zeisel_0.0_1_recon.npy --output otherresults/dca/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv