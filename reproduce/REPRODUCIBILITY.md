# Reproduce 

Scripts to reproduce results obtained in the manuscript

## Preprocess benchmarks

### Option 1 (Recommended): directly use preproceed data

```shell
cd Data
tar zxvf benchmarkData.tar.gz 
```

There are four datasets: Chung, Kolodziejczyk, Klein, Zeisel

### Option 2: regenerate preproceed data

#### 1. generating usage csv

Take Dataset Chung for example.

```shell
python Preprocessing_benchmark.py --inputfile /Users/wangjue/workspace/scGNN/Data/benchmarkData/Chung/T2000_expression.txt --outputfile /Users/wangjue/workspace/scGNN/Chung.csv --split space --cellheadflag False --cellcount 317

python Preprocessing_benchmark.py --inputfile /Users/wangjue/workspace/scGNN/Data/benchmarkData/Kolodziejczyk/T2000_expression.txt --outputfile /Users/wangjue/workspace/scGNN/Kolodziejczyk.csv --split space --cellheadflag False --cellcount 704

python Preprocessing_benchmark.py --inputfile /Users/wangjue/workspace/scGNN/Data/benchmarkData/Klein/T2000_expression.txt --outputfile /Users/wangjue/workspace/scGNN/Klein.csv --split space --cellheadflag False --cellcount 2717

python Preprocessing_benchmark.py --inputfile /Users/wangjue/workspace/scGNN/Data/benchmarkData/Zeisel/T2000_expression.txt --outputfile /Users/wangjue/workspace/scGNN/Zeisel.csv --split space --cellheadflag False --cellcount 3005
```

#### 2. generating sparse coding under data/

```shell
python Preprocessing_main.py --expression-name Chung --featureDir /Users/wangjue/workspace/scGNN/
```

## Clustering on Benchmarks

```
python3 -W ignore main_benchmark.py --datasetName Chung --benchmark /Users/wangjue/workspace/scGNN/Data/benchmarkData/Chung/Chung_cell_label.csv --LTMGDir /Users/wangjue/workspace/scGNN/Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu/ --debuginfo  
```

## Imputation on Benchmarks

Default: 10% of the non-zeros are flipped

```
python3 -W ignore main_benchmark.py --datasetName Chung --benchmark /Users/wangjue/workspace/scGNN/Data/benchmarkData/Chung/Chung_cell_label.csv --LTMGDir /Users/wangjue/workspace/scGNN/Data/benchmarkData/Chung/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu/ --debuginfo --imputeMode
```
