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
python -W ignore louvain.py --input othermethods/magic/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/magic/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
python -W ignore louvain.py --input othermethods/magic/12.Klein_0.0_1_recon.npy --output otherresults/magic/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
python -W ignore louvain.py --input othermethods/magic/13.Zeisel_0.0_1_recon.npy --output otherresults/magic/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/dca/9.Chung_0.0_1_recon.npy --output otherresults/dca/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/dca/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/dca/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/dca/12.Klein_0.0_1_recon.npy --output otherresults/dca/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/dca/13.Zeisel_0.0_1_recon.npy --output otherresults/dca/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/deepimpute/9.Chung_0.0_1_recon.npy --output otherresults/deepimpute/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/deepimpute/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/deepimpute/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/deepimpute/12.Klein_0.0_1_recon.npy --output otherresults/deepimpute/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/deepimpute/13.Zeisel_0.0_1_recon.npy --output otherresults/deepimpute/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/netNMFsc/9.Chung_0.0_1_recon.npy --output otherresults/netNMFsc/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/netNMFsc/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/netNMFsc/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/netNMFsc/12.Klein_0.0_1_recon.npy --output otherresults/netNMFsc/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/netNMFsc/13.Zeisel_0.0_1_recon.npy --output otherresults/netNMFsc/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/saucie/9.Chung_0.0_1_recon.npy --output otherresults/saucie/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/saucie/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/saucie/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/saucie/12.Klein_0.0_1_recon.npy --output otherresults/saucie/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/saucie/13.Zeisel_0.0_1_recon.npy --output otherresults/saucie/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/saver/9.Chung_0.0_1_recon.npy --output otherresults/saver/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/saver/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/saver/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/saver/12.Klein_0.0_1_recon.npy --output otherresults/saver/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/saver/13.Zeisel_0.0_1_recon.npy --output otherresults/saver/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/scIGANs/9.Chung_0.0_1_recon.npy --output otherresults/scIGANs/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/scIGANs/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/scIGANs/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/scIGANs/12.Klein_0.0_1_recon.npy --output otherresults/scIGANs/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/scIGANs/13.Zeisel_0.0_1_recon.npy --output otherresults/scIGANs/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/scimpute/9.Chung_0.0_1_recon.npy --output otherresults/scimpute/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/scimpute/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/scimpute/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/scimpute/12.Klein_0.0_1_recon.npy --output otherresults/scimpute/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/scimpute/13.Zeisel_0.0_1_recon.npy --output otherresults/scimpute/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv

# python -W ignore louvain.py --input othermethods/scvi/9.Chung_0.0_1_recon.npy --output otherresults/scvi/9.txt --benchmark /home/jwang/data/scData/9.Chung/Chung_cell_label.csv
# python -W ignore louvain.py --input othermethods/scvi/11.Kolodziejczyk_0.0_1_recon.npy --output otherresults/scvi/11.txt --benchmark /home/jwang/data/scData/11.Kolodziejczyk/Kolodziejczyk_cell_label.csv
# python -W ignore louvain.py --input othermethods/scvi/12.Klein_0.0_1_recon.npy --output otherresults/scvi/12.txt --benchmark /home/jwang/data/scData/12.Klein/Klein_cell_label.csv
# python -W ignore louvain.py --input othermethods/scvi/13.Zeisel_0.0_1_recon.npy --output otherresults/scvi/13.txt --benchmark /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv
