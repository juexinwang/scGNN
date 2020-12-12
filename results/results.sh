#! /bin/bash
######################### Batch Headers #########################
#SBATCH -A xulab
#SBATCH -p Lewis,BioCompute               # use the BioCompute partition Lewis,BioCompute
#SBATCH -J Louvain
#SBATCH -o results-%j.out           # give the job output a custom name
#SBATCH -t 2-00:00                  # two days time limit
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores (AKA tasks)
#SBATCH --mem=128G
#################################################################
module load miniconda3
source activate conda_R

python -W ignore results_tmp.py --inputOri othermethods/saucie/12.Klein_0.0_1_recon.npy --input otherresults/saucie/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
python -W ignore results_tmp.py --inputOri othermethods/saucie/13.Zeisel_0.0_1_recon.npy --input otherresults/saucie/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

python -W ignore results_tmp.py --inputOri othermethods/scvi/12.Klein_0.0_1_recon.npy --input otherresults/scvi/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
python -W ignore results_tmp.py --inputOri othermethods/scvi/13.Zeisel_0.0_1_recon.npy --input otherresults/scvi/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

python -W ignore results_tmp.py --inputOri othermethods/netNMFsc/12.Klein_0.0_1_recon.npy --input otherresults/netNMFsc/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
python -W ignore results_tmp.py --inputOri othermethods/netNMFsc/13.Zeisel_0.0_1_recon.npy --input otherresults/netNMFsc/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

python -W ignore results_tmp.py --inputOri othermethods/scIGANs/12.Klein_0.0_1_recon.npy --input otherresults/scIGANs/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
python -W ignore results_tmp.py --inputOri othermethods/scIGANs/13.Zeisel_0.0_1_recon.npy --input otherresults/scIGANs/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv
