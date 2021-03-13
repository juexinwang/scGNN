# Tutorial of scGNN   

## About:

__scGNN__ (**s**ingle **c**ell **g**raph **n**eural networks) provides a hypothesis-free deep learning framework for scRNA-Seq analyses. This framework formulates and aggregates cell-cell relationships with graph neural networks and models heterogeneous gene expression patterns using a left-truncated mixture Gaussian model. scGNN integrates three iterative multi-modal autoencoders and outperforms existing tools for gene imputation and cell clustering on four benchmark scRNA-Seq datasets.

## Installation:

Installation Tested on Ubuntu 16.04, CentOS 7, MacOS catalina with Python 3.6.8 on one NVIDIA RTX 2080Ti GPU.

### From Source:

Start by grabbing this source codes:

```bash
git clone https://github.com/juexinwang/scGNN.git
cd scGNN
```

### Option 1 : (Recommended) Use python virutal enviorment with conda（<https://anaconda.org/>）

```shell
conda create -n scgnnEnv python=3.6.8 pip
conda activate scgnnEnv
pip install -r requirements.txt
```

If want to use LTMG (**Recommended** but Optional, will takes extra time in data preprocessing):
```shell
conda install r-devtools
conda install -c cyz931123 r-scgnnltmg
```

### Option 2 : Direct install individually

Need to install R packages first, tested on R >=3.6.1:

In R command line:

```R
install.packages("devtools")
library(devtools)
install_github("BMEngineeR/scGNNLTMG")
```

Then install all python packages in bash.

```bash
pip install -r requirements.txt
```

### Option 3: Use Docker

Download and install [docker](https://www.docker.com/products/docker-desktop).

Pull docker image **gongjt057/scgnn** from the [dockerhub](https://hub.docker.com/). Please beware that this image is huge for it includes all the environments. To get this docker image as base image type the command as shown in the below:

```bash
docker pull gongjt057/scgnn:code
```

Type `docker images` to see the list of images you have downloaded on your machine. If **gongjt057/scgnn** in the list, download it successfully.

Run a container from the image.

```bash
docker run -it gongjt057/scgnn:code /bin/bash
cd /scGNN/scGNN
```

Then you can proceed to the next step.

## Preprocess

Datasets should be preprocessed to proceed. scGNN accepts scRNA-seq data format: CSV and 10X

### 1. Prepare datasets 

#### CSV format

Take an example of Alzheimer’s disease datasets ([GSE138852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138852)) analyzed in the manuscript.

```shell
mkdir GSE138852
wget -P GSE138852/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138852/suppl/GSE138852_counts.csv.gz
```

#### 10X format

Take an example of [liver cellular landscape study](https://data.humancellatlas.org/explore/projects/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674/expression-matrices?catalog=dcp1) from human cell atlas(<https://data.humancellatlas.org/>). Click the download link of 'homo_sapiens.mtx.zip' in the page, and get 4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip. (It looks like they does not provide direct download link anymore)

```shell
mkdir liver
cd liver
mv ~/Download/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip .
unzip 4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
cd ..
```

### 2. Preprocess input files for scGNN

Program ***PreprocessingscGNN.py*** is used to generate input files for scGNN. This step generates Use_expression.csv (preprocessed file) and gets discretized regulatory signals as ltmg.csv from Left-Trunctruncated-Mixed-Gaussian(LTMG) model (Optional but recommended).  

In ***PreprocessingscGNN.py***, usually these parameters are used:

#### Required

- **datasetName** defines the raw file name of scRNA-Seq
- **datasetDir** folder of the raw file
- **LTMGDir** Ouput folder of the LTMG output
- **filetype** defines file type (CSV or 10X(default))

#### Optional 

- **transform** defines the type of transformation, default is logarithm transformation. In the implementation, we use log(x+1), x is the original expression level.
- **cellRatio** defines the maximum ratio of zeros in cells. Default is 0.99. This parameter filters out cells with more than 99% genes that are zeros. 
- **geneRatio** defines the maximum ratio of zeros in genes. Default is 0.99. This parameter filters out genes with more than 99% genes that are zeros.
- **geneCriteria** defines which criteria to select most variant genes, default is variance.
- **geneSelectnum** selects a number of most variant genes. The default gene number is 2000. 

The running time of ***PreprocessingscGNN.py*** is dependent on the cell number and gene numbers selected. It takes ~10 minutes (GSE138852) and ~13 minutes (liver) to generate the files needed.

#### Example:

#### CSV format

Cell/Gene filtering without inferring LTMG:
```shell
python -W ignore PreprocessingscGNN.py --datasetName GSE138852_counts.csv.gz --datasetDir GSE138852/ --LTMGDir GSE138852/ --filetype CSV --geneSelectnum 2000
```

(Optional) Cell/Gene filtering and inferring LTMG:
```shell
python -W ignore PreprocessingscGNN.py --datasetName GSE138852_counts.csv.gz --datasetDir GSE138852/ --LTMGDir GSE138852/ --filetype CSV --geneSelectnum 2000 --inferLTMGTag
```

#### 10X format

Cell/Gene filtering without inferring LTMG:
```shell
python -W ignore PreprocessingscGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --datasetDir liver/ --LTMGDir liver/ --geneSelectnum 2000 sparseOut
```

(Optional) Cell/Gene filtering and inferring LTMG:
```shell
python -W ignore PreprocessingscGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --datasetDir liver/ --LTMGDir liver/ --geneSelectnum 2000 --inferLTMGTag
```

### 3. Run scGNN

Program ***scGNN.py*** is the main entrance of scGNN to impute and clustering. There are quite a few parameters to define to meet users' requirements.

#### Required

- **datasetName** defines the folder of scRNA-Seq
- **LTMGDir** defines folder of the preprocessed LTMG output
- **outputDir** Output folder of the results

#### Clustering related

- **clustering-method** Clustering method on identifying celltypes from the embedding. Default is LouvainK: use Louvain to determine the number of the clusters and then use K-means. Supporting clustering type: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB
- **n-clusters** predefines the number of clusters, it only used for clustering methods need a number of clusters input as KNN
- **maxClusterNumber** defines the maximum number of cluster allowed, default is 30. This parameter prevents extreme cases that too many clusters identified by Louvian clustering
- **minMemberinCluster** defines the minimum number of cells in a cluster, default is 5. This parameter prevents extreme cases that too many clusters identified by Louvain clustering.
- **resolution** controls the number of clusters identified by Louvain clustering. This parameter can be set between 0.4 and 1.2 in most cases. According to results on benchmarks, we set default 'auto'.

#### Optional: Hyperparameters

- **EM-iteration** defines the number of iteration, default is 10 
- **Regu-epochs** defines epochs in Feature Autoencoder initially, default is 500
- **EM-epochs** defines epochs in Feature Autoencoder in the iteration, default is 200
- **cluster-epochs** defines epochs in the Cluster Autoencoder, default is 200
- **k** is k of the K-Nearest-Neighour Graph
- **knn-distance** distance type of building K-Nearest-Neighour Graph, supported type: euclidean/cosine/correlation (default: euclidean)
- **GAEepochs** Number of epochs to train in Graph Autoencoder

#### Optional: Performance

- **quickmode** whether or not to bypass the Cluster Autoencoder.
- **useGAEembedding** whether use Graph Autoencoder
- **regulized-type** is the regularized type: noregu/LTMG, default is to use LTMG
- **alphaRegularizePara** alpha in the manuscript, the intensity of the regularizer
- **EMregulized-type** defines the imputation regularizer type:noregu/Graph/Celltype, default: Celltype
- **gammaImputePara** defines the intensity of LTMG regularizer in Imputation
- **graphImputePara** defines the intensity of graph regularizer in Imputation
- **celltypeImputePara** defines the intensity of celltype regularizer in Imputation
- **L1Para** defines the intensity of L1 regularizer, default: 1.0 
- **L2Para** defines the intensity of L2 regularizer, defualt: 0.0
- **saveinternal** whether output internal results for debug usage

#### Optional: Speed

- **no-cuda** defines devices in usage. Default is using GPU, add --no-cuda in command line if you only have CPU.
- **coresUsage** defines how many cores can be used. default: 1. Change this value if you want to use more.

#### Example: 

#### CSV format

For CSV format, we need add **--nonsparseMode**

Without LTMG:
```bash
python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode
```

(Optional) Using LTMG:
```bash
python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./ --LTMGDir ./ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode --regulized-type LTMG
```

#### 10X format

Without LTMG:
```bash
python -W ignore scGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --datasetDir liver/ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode
```

(Optional) Using LTMG:
```bash
python -W ignore scGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --LTMGDir liver/ --datasetDir liver/ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --regulized-type LTMG
```

On these demo dataset using single cpu, the running time of demo codes is ~33min/26min. User should get exact same results as paper shown with full running time on single cpu for ~6 hours. If user wants to use multiple CPUs, parameter **--coresUsage** can be set as **all** or any number of cores the machine has.

## Check Results

In outputdir now, we have four output files.

- ***_recon.csv**:        Imputed gene expression matrix. Row as gene, col as cell. First row as gene name, First col as the cell name. 

- ***_embedding.csv**:    Learned embedding (features) for clustering. Row as cell, col as embeddings. First row as the embedding names (no means). First col as the cell name.

- ***_graph.csv**:        Learned graph edges of the cell graph in tuples: nodeA,nodeB,weights. First row as the name.

- ***_results.txt**:      Identified cell types. First row as the name. 

For a complete list of options provided by scGNN:

```bash
python scGNN.py --help
```
