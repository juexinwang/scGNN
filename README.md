# scGNN

single cell Graph Neural Networks

About:
----------
Graph Neural Network for Single Cell Impute and celltype identification. 

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
    pip install munkres

R integration:

    R >=3.6.2
    install.packages("devtools")
    install.packages("igraph")
    library(devtools)
    install_github("BMEngineeR/scGNNLTMG")

Example:
---------
Example data:
After filtering: 9760 cells 13052 genes, finally select 2000 genes
https://data.humancellatlas.org/project-assets/project-matrices/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
30K liver cells (10X)

1. Generating Use_expression.csv (preprocessed file) and ltmg.csv (ltmg)

    `python3 -W ignore PreprocessingscGNN.py --datasetName e7448a34-b33d-41de-b422-4c09bfeba96b.mtx --datasetDir /storage/htc/joshilab/wangjue/10x/6/ --LTMGDir /storage/htc/joshilab/wangjue/10x/6/`

2. Run scGNN

    `module load miniconda3`
    
    `source activate conda_R`

    `python3 -W ignore scGNN.py --datasetName e7448a34-b33d-41de-b422-4c09bfeba96b.mtx --LTMGDir /storage/htc/joshilab/wangjue/10x/6/ --outputDir outputdir/`

3. Check Results
    
    In outputdir now, we have four output files. May update later?
    
    *_recon.csv:        Imputed gene expression matrix. Row as gene, col as cell. First row as gene name, First col as the cell name. 

    *_embedding.csv:    Learned embedding (features) for clustering. Row as cell, col as embeddings. First row as the embedding names (no means). First col as the cell name

    *_graph.csv:        Learned graph edges of the cell graph in tuples: nodeA,nodeB,weights. First row as the name.

    *_results.txt:      Identified cell types. First row as the name. 


Notes for Cluster Running Benchmark: (main_benchmark.py) Here for eproducibility.
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

    experiment_2_g_e.sh *

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
