# Reproduce 

Scripts to reproduce results obtained in the manuscript

## Preprocess benchmarks

### Option 1 (Recommended): directly use preproceed data

```shell
cd data
tar zxvf data.tar.gz 
```

There are four datasets: Chung, Kolodziejczyk, Klein, Zeisel

### Option 2: regenerate preproceed data

#### 1. generating usage csv

Take Dataset Chung for example.

```shell
python Preprocessing_scFile.py --inputfile /folder/scGNN/data/9.Chung/T2000_expression.txt --outputfile /home/wangjue/biodata/scData/9.Chung.csv --split space --cellheadflag False
```

#### 2. generating sparse coding under data/

```shell
python Preprocessing_main.py --expression-name 9.Chung
```

## Clustering on Benchmarks

```
python3 -W ignore main_benchmark.py --datasetName 9.Chung --benchmark data/9.Chung/Chung_cell_label.csv --npyDir output/  
```

## Imputation on Benchmarks

Default: 10% of the non-zeros are flipped

```
python3 -W ignore main_benchmark.py --datasetName 9.Chung --benchmark data/9.Chung/Chung_cell_label.csv --npyDir output/ --imputeMode
```
