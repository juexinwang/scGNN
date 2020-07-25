# scGNN   

## About:

__scGNN__ (**s**ingle **c**ell **g**raph **n**eural networks) provides a hypothesis-free deep learning framework for scRNA-Seq analyses. This framework formulates and aggregates cell-cell relationships with graph neural networks and models heterogeneous gene expression patterns using a left-truncated mixture Gaussian model. scGNN integrates three iterative multi-modal autoencoders and outperforms existing tools for gene imputation and cell clustering on four benchmark scRNA-Seq datasets.

## Installation:

Tested on Ubuntu 16.04 and CentOS 7 with Python 3.6.8

### From Source:

Start by grabbing this source codes:

```bash
git clone https://github.com/scgnn/scGNN.git
cd scGNN
```

### Option 1 : (Recommended) Use python virutal enviorment with conda（<https://anaconda.org/>）

```shell
conda create -n scgnnEnv python=3.6.8 pip
conda activate scgnnEnv
conda install r-devtools
conda install -c r r-igraph
conda install -c cyz931123 r-scgnnltmg
pip install -r requirements.txt
```

### Option 2 : Direct install individually

Need to install R packages first, tested on R >=3.6.1:

In R command line:

```R
install.packages("devtools")
install.packages("igraph")
library(devtools)
install_github("BMEngineeR/scGNNLTMG")
```

Then install all python packages in bash.

```bash
pip install -r requirements.txt
```

### Option 3: Use Docker 
    
#TODO

## Quick Start:

scGNN accepts scRNA-seq data format: CSV and 10X

### 1. Prepare datasets 

#### CSV format

Take example of Alzheimer’s disease datasets ([GSE138852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138852)) analyzed in the manuscript.

```shell
mkdir GSE138852
wget -P GSE138852/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138852/suppl/GSE138852_counts.csv.gz
```

#### 10X format

Take example of [liver cellular landscape study](https://data.humancellatlas.org/explore/projects/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674) from human cell atlas(<https://data.humancellatlas.org/>)

```shell
mkdir liver
wget -P liver https://data.humancellatlas.org/project-assets/project-matrices/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
cd liver
unzip 4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
cd ..
```

### 2. Preprocess input files 

This step generates Use_expression.csv (preprocessed file) and get discretirized regulatory signals as ltmg.csv from Left-Trunctruncated-Mixed-Gaussian(LTMG) model (Optional but recommended).  

In preprocessing, parameters are used:

- **filetype** defines file type (CSV or 10X(default))  
- **geneSelectnum** selects number of most variant genes. The default gene number is 2000 

The running time is depended with the cell number and gene numbers selected. It takes ~20 minutes (GSE138852) and ~28 minitues (liver) to generate the files needed.

#### CSV format

```shell
python3 -W ignore PreprocessingscGNN.py --datasetName GSE138852_counts.csv.gz --datasetDir GSE138852/ --LTMGDir GSE138852/ --filetype CSV --geneSelectnum 2000
```

#### 10X format

```shell
python3 -W ignore PreprocessingscGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --datasetDir liver/ --LTMGDir liver/ --geneSelectnum 2000
```

### 3. Run scGNN 

We takes example of analysis in GSE138852. Here we use parameters to demo purposes:

- **EM-iteration** defines number of iteration, default is 10, here we set as 2. 
- **quickmode** for bypassing cluster autoencoder.
- **Regu-epochs** defines epocs in feature autoencoder, default is 500, here we set as 50.
- **EM-epochs** defines epocs in feature autoencoder in the iteration, default is 200, here we set as 20.

If you want to reproduce results in the manuscript, do not use these parameters. 

#### CSV format

For CSV format, we need add **--nonsparseMode**

```bash
python3 -W ignore scGNN.py --datasetName GSE138852 --LTMGDir ./ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode
```

#### 10X format

```bash
python3 -W ignore scGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --LTMGDir liver/ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode
```

On these demo dataset using single cpu, the running time of demo codes is ~33min/26min. The full running time is ~6 hours.

### 4. Check Results
    
In outputdir now, we have four output files.
    
- ***_recon.csv**:        Imputed gene expression matrix. Row as gene, col as cell. First row as gene name, First col as the cell name. 

- ***_embedding.csv**:    Learned embedding (features) for clustering. Row as cell, col as embeddings. First row as the embedding names (no means). First col as the cell name.

- ***_graph.csv**:        Learned graph edges of the cell graph in tuples: nodeA,nodeB,weights. First row as the name.

- ***_results.txt**:      Identified cell types. First row as the name. 

For a complete list of options provided by scGNN:

```
python scGNN.py --help
```

More information can be checked at [tutorial](https://github.com/scgnn/scGNN/tree/master/tutorial).

## Reference:

1. VAE <https://github.com/pytorch/examples/tree/master/vae>
2. GAE <https://github.com/tkipf/gae/tree/master/gae>
3. scVI-reproducibility <https://github.com/romain-lopez/scVI-reproducibility>
4. LTMG <https://academic.oup.com/nar/article/47/18/e111/5542876>
