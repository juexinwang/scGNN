# scGNN

single cell Graph Neural Networks

About:
----------
Graph Neural Network for Single Cell Impute and celltype classification 

Reqirements: (May not sufficient)
----------
Tested on Ubuntu 16.04 and CentOS 7 with Python 3.6.8

    pip install numpy
    pip install torch
    pip install networkx
    pip install matplotlib
    pip install pandas
    pip install seaborn
    pip install umap-learn
    pip install community
    pip install rpy2
    pip install node2vec
    pip install munkres

R integration:

    R >=3.6.2
    install.packages("devtools")
    library(devtools)
    install_github("dgrun/FateID")
    install_github("dgrun/RaceID3_StemID2_package")

Notes for Cluster Running:
---------
module load miniconda3
conda create -n my_environment python=3.7
source activate my_environment

* for using R support: geneLouvianCluster.py, otherwise we do not use rpy2
experiment_1_g_e.sh
experiment_1_g_f.sh
experiment_1_n_e.sh
experiment_1_n_f.sh
experiment_2_g_e_AffinityPropagation.sh
experiment_2_g_e_AgglomerativeClustering.sh
experiment_2_g_e_Birch.sh
experiment_2_g_e_KMeans.sh
experiment_2_g_e.sh *
experiment_2_g_e_SpectralClustering.sh
experiment_2_g_f_AffinityPropagation.sh
experiment_2_g_f_AgglomerativeClustering.sh
experiment_2_g_f_Birch.sh
experiment_2_g_f_KMeans.sh
experiment_2_g_f.sh *
experiment_2_g_f_SpectralClustering.sh
experiment_2_n_e.sh *
experiment_2_n_f.sh *

Reference:
---------

1. VAE <https://github.com/pytorch/examples/tree/master/vae>
2. GAE <https://github.com/tkipf/gae/tree/master/gae>

Contact:
---------
Juexin Wang wangjue@missouri.edu
