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
    pip install dask[dataframe]

R integration:

    R >=3.6.2
    install.packages("devtools")
    library(devtools)
    install_github("dgrun/FateID")
    install_github("dgrun/RaceID3_StemID2_package")
    install_github("BMEngineeR/scGNNLTMG")

*** Notes for casestudy: (scGNN.py) Temporary
---------
Example data:
After filtering: 9760 cells 13052 genes, finally select 2000 genes
https://data.humancellatlas.org/project-assets/project-matrices/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
30K liver cells (10X)

1. Generating Use_expression.csv (preprocessed file) and ltmg.csv (ltmg)

    python3 -W ignore PreprocessingscGNN.py --datasetName e7448a34-b33d-41de-b422-4c09bfeba96b.mtx --datasetDir /storage/htc/joshilab/wangjue/10x/6/ --LTMGDir /storage/htc/joshilab/wangjue/10x/6/

2. Run scGNN

    module load miniconda3
    source activate conda_R
    python3 -W ignore scGNN.py --datasetName e7448a34-b33d-41de-b422-4c09bfeba96b.mtx --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --npyDir outputdir/

3. Check Results
    
    In outputdir now, we have four output files: Need to change to csv files later：
    
    *_recon.csv:        imputed matrix. Row as cell, col as gene 

    *_embedding.csv:    learned embedding (features) for clustering. Row as cell, col as embeddings

    *_graph.csv:        learned graph edges of the cell graph: node1,node2,weights

    *_results.txt:      groups of cells identified. 


Notes for Cluster Running Benchmark: (main_benchmark.py) May be deleted later
---------
module load miniconda3
conda create -n my_environment python=3.7
source activate my_environment

Preprocess benchmarks:

# 1. generating usage csv

python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/10.Usoskin/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/10.Usoskin.csv --cellcount 622 --genecount 2000 --split space --cellheadflag False

# 2. generating sparse coding under data/
python Preprocessing_main.py --expression-name 10.Usoskin


Now We totally have 63 methods in testing:
* for using R support: geneLouvianCluster.py, otherwise we do not use rpy2

    experiment_1_g_b.sh
    experiment_1_g_e.sh
    experiment_1_g_f.sh
    experiment_1_r_b.sh
    experiment_1_r_e.sh
    experiment_1_r_f.sh
    experiment_1_n_b.sh
    experiment_1_n_e.sh
    experiment_1_n_f.sh
    experiment_2_g_b_AffinityPropagation.sh
    experiment_2_g_b_AgglomerativeClustering.sh
    experiment_2_g_b_Birch.sh
    experiment_2_g_b_KMeans.sh
    experiment_2_g_b_SpectralClustering.sh
    experiment_2_g_b.sh *
    experiment_2_g_e_AffinityPropagation.sh
    experiment_2_g_e_AgglomerativeClustering.sh
    experiment_2_g_e_Birch.sh
    experiment_2_g_e_KMeans.sh
    experiment_2_g_e_SpectralClustering.sh
    experiment_2_g_e.sh *   
    experiment_2_g_f_AffinityPropagation.sh
    experiment_2_g_f_AgglomerativeClustering.sh
    experiment_2_g_f_Birch.sh
    experiment_2_g_f_KMeans.sh
    experiment_2_g_f_SpectralClustering.sh
    experiment_2_g_f.sh *
    experiment_2_r_b_AffinityPropagation.sh
    experiment_2_r_b_AgglomerativeClustering.sh
    experiment_2_r_b_Birch.sh
    experiment_2_r_b_KMeans.sh
    experiment_2_r_b_SpectralClustering.sh
    experiment_2_r_b.sh *
    experiment_2_r_e_AffinityPropagation.sh
    experiment_2_r_e_AgglomerativeClustering.sh
    experiment_2_r_e_Birch.sh
    experiment_2_r_e_KMeans.sh
    experiment_2_r_e_SpectralClustering.sh
    experiment_2_r_e.sh *   
    experiment_2_r_f_AffinityPropagation.sh
    experiment_2_r_f_AgglomerativeClustering.sh
    experiment_2_r_f_Birch.sh
    experiment_2_r_f_KMeans.sh
    experiment_2_r_f_SpectralClustering.sh
    experiment_2_r_f.sh * 
    experiment_2_n_b_AffinityPropagation.sh
    experiment_2_n_b_AgglomerativeClustering.sh
    experiment_2_n_b_Birch.sh
    experiment_2_n_b_KMeans.sh
    experiment_2_n_b_SpectralClustering.sh
    experiment_2_n_b.sh *   
    experiment_2_n_e_AffinityPropagation.sh
    experiment_2_n_e_AgglomerativeClustering.sh
    experiment_2_n_e_Birch.sh
    experiment_2_n_e_KMeans.sh
    experiment_2_n_e_SpectralClustering.sh
    experiment_2_n_e.sh *    
    experiment_2_n_f_AffinityPropagation.sh
    experiment_2_n_f_AgglomerativeClustering.sh
    experiment_2_n_f_Birch.sh
    experiment_2_n_f_KMeans.sh
    experiment_2_n_f_SpectralClustering.sh
    experiment_2_n_f.sh *   

1. Generating shells for sbatch: This will generate lots of shell files!

    python generatingMethodsBatchshell.py
    python generatingMethodsBatchshell.py --imputeMode

2. Submit shells in cluster (Lewis in University of Missouri):

    submitCluster_Celltype.sh
    submitCluster_Impute.sh

3. Get results in cluster

    3.1 Generating results shells and Submit scripts to cluster, we use batch mode here
        cd results
        bash submitCluster_Result_Celltype.sh
        bash submitCluster_Result_Impute.sh
        
    3.2 On cluster, store the job information as jobinfo.txt

    3.3 (on Localmachine)Parsing results when ready:
        python summary_cmd.py 

Reference:
---------

1. VAE <https://github.com/pytorch/examples/tree/master/vae>
2. GAE <https://github.com/tkipf/gae/tree/master/gae>
3. scVI-reproducibility <https://github.com/romain-lopez/scVI-reproducibility>

Contact:
---------
Juexin Wang wangjue@missouri.edu
