# scGNN

single cell Graph Neural Networks

## About:

Graph Neural Network for Single Cell scRNA Imputation and celltype identification. 

## Reqirements:

Tested on Ubuntu 16.04 and CentOS 7 with Python 3.6.8

Option 1: Direct individual install

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

Option 2: simply run ```pip install -r requirements.txt``` to install all the necessary packages.

Option 3: Use Docker #TODO

## Example:

Accepting scRNA format: 10X and CSV
Example data:
After filtering: 9760 cells 13052 genes, finally select 2000 genes
https://data.humancellatlas.org/project-assets/project-matrices/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip

30K liver cells (10X)

1. Generating Use_expression.csv (preprocessed file) and ltmg.csv (ltmg)
TODO: ltmgDir
- CSV

    `python3 -W ignore PreprocessingscGNN.py --datasetName e7448a34-b33d-41de-b422-4c09bfeba96b.mtx --datasetDir /storage/htc/joshilab/wangjue/10x/6/ --LTMGDir /storage/htc/joshilab/wangjue/10x/6/ --filetype CSV`

- 10X

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


## Notes for Cluster Running Benchmark: (main_benchmark.py) Here for eproducibility.
---------
    module load miniconda3
    conda create -n my_environment python=3.7
    source activate my_environment

Preprocess benchmarks:

 1. generating usage csv

    python Preprocessing_scFile.py --inputfile /home/wangjue/biodata/scData/allBench/10.Usoskin/T2000_UsingOriginalMatrix/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/10.Usoskin.csv --cellcount 622 --genecount 2000 --split space --cellheadflag False

2. generating sparse coding under data/
    python Preprocessing_main.py --expression-name 10.Usoskin

Or directly unzip data folder ```gunzip data```

Now We totally have 4 dropout ratio in testing imputation (0.1,0.3,0.6,0.9):

1. Generating job scripts for each of the benchmark datsets:

    ```python generating_Impute_0.1-0.8.py```

2. Submit job scripts in cluster (HPC):

    ```submitCluster_impute_0.1-0.8.sh```

3. Get results when jobs finished

    ```cd results``` 
    ```bash results_impute_explore_0.3.sh```

(Optional): Check MAGIC results: 
    ```cd otherresults``` 
    ```bash results_impute_explore_0.3.sh```

Identify celltypes

1. Generating job scripts for each of the benchmark datsets:

    ```python generating_celltype.py```

2. Submit job scripts in cluster (HPC):

    ```submitCluster_celltype.sh```

3. Get results in npyG2E_LK_1 when jobs finished
 
    ```ls npyG2E_LK_1/*_benchmark.txt```


## Reference:

1. VAE <https://github.com/pytorch/examples/tree/master/vae>
2. GAE <https://github.com/tkipf/gae/tree/master/gae>
3. scVI-reproducibility <https://github.com/romain-lopez/scVI-reproducibility>

Note:
DeepImpute:
https://github.com/lanagarmire/deepimpute
Change to https://github.com/lanagarmire/deepimpute/blob/master/deepimpute/multinet.py
from tensorflow.keras.callbacks import EarlyStopping

## Contact:

Juexin Wang wangjue@missouri.edu
