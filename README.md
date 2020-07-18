# scGNN   

## About:

__scGNN__ (**s**ingle **c**ell **g**raph **n**eural networks) provides a hypothesis-free deep learning framework for scRNA-Seq analyses. This framework formulates and aggregates cell-cell relationships with graph neural networks and models heterogeneous gene expression patterns using a left-truncated mixture Gaussian model. scGNN integrates three iterative multi-modal autoencoders and outperforms existing tools for gene imputation and cell clustering on four benchmark scRNA-Seq datasets.

## Installation:

Tested on Ubuntu 16.04 and CentOS 7 with Python 3.6.8

### Option 1: (Recommended) Use python virutal enviorment with conda（<https://anaconda.org/>）

```shell
conda create -n scgnnEnv python=3.6.8 pip
conda activate scgnnEnv
conda install r-devtools
conda install -c r r-igraph
pip install -r requirements.txt
```

### Option 2 : Direct install individually

    Installing R packages, tested on R >=3.6.2:
    In R command line:

```R
install.packages("devtools")
install.packages("igraph")
library(devtools)
install_github("BMEngineeR/scGNNLTMG")
```

    Install all python packages.

```bash
pip install -r requirements.txt
```

### Option 3: Use Docker 
    
    #TODO

## Quick Start:

scGNN accepts scRNA-seq data format: CSV and 10X

1. Prepare datasets

- CSV format
    Take example of Alzheimer’s disease datasets （GSE138852） analyzed in the manuscript.
    ```shell
    mkdir GSE138852
    wget -P GSE138852/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138852/suppl/GSE138852_counts.csv.gz
    ```

- 10X format
    Take example of 30K liver cells from human cell atlas
    ```shell
    mkdir liver
    wget -P liver https://data.humancellatlas.org/project-assets/project-matrices/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
    ```
    
2. Preprocess input files and get discretirized regulatory signals from Left-Trunctruncated-Mixed-Gaussian(LTMG) model (Optional but recommended). This step will generate Use_expression.csv (preprocessed file) and ltmg.csv (from LTMG). 
In preprocessing, paramter **geneSelectnum** selects number of most variant genes. The default gene number is 2000.  

- CSV format

```shell
python3 -W ignore PreprocessingscGNN.py --datasetName GSE138852_counts.csv --datasetDir /folder/GSE138852/ --LTMGDir /folder/GSE138852/ --filetype CSV --geneSelectnum 2000
```

- 10X format

```shell
python3 -W ignore PreprocessingscGNN.py --datasetName e7448a34-b33d-41de-b422-4c09bfeba96b.mtx --datasetDir /folder/liver/ --LTMGDir /folder/liver/ --geneSelectnum 2000
```

3. Run scGNN. We takes example of analysis in GSE138852. Here wer use parameters to demo purposes: 
    - **EM-iteration** defines number of iteration, default is 10, here we set as 2. 
    - **quickmode** for bypassing cluster autoencoder. 
    
    If you want to reproduce results in the manuscript, not using these two parameters. 

    ```bash
    python3 -W ignore scGNN.py --datasetName GSE138852_counts.csv --LTMGDir /folder/GSE138852/ --outputDir outputdir/ --EM-iteration 2 --quickmode
    ```

4. Check Results
    
    In outputdir now, we have four output files.
    
    *_recon.csv:        Imputed gene expression matrix. Row as gene, col as cell. First row as gene name, First col as the cell name. 

    *_embedding.csv:    Learned embedding (features) for clustering. Row as cell, col as embeddings. First row as the embedding names (no means). First col as the cell name.

    *_graph.csv:        Learned graph edges of the cell graph in tuples: nodeA,nodeB,weights. First row as the name.

    *_results.txt:      Identified cell types. First row as the name. 

## Reference:

1. VAE <https://github.com/pytorch/examples/tree/master/vae>
2. GAE <https://github.com/tkipf/gae/tree/master/gae>
3. scVI-reproducibility <https://github.com/romain-lopez/scVI-reproducibility>
4. LTMG <https://academic.oup.com/nar/article/47/18/e111/5542876>
