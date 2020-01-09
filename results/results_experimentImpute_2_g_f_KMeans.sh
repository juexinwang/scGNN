python3 -W ignore main.py --datasetName MMPbasal  --npyDir ../npyImputeG2F_KMeans/
python3 -W ignore main.py --datasetName MMPbasal --discreteTag  --npyDir ../npyImputeG2F_KMeans/
python3 -W ignore main.py --datasetName MMPbasal_LTMG  --npyDir ../npyImputeG2F_KMeans/

python3 -W ignore main.py --datasetName MMPbasal_all  --npyDir ../npyImputeG2F_KMeans/
python3 -W ignore main.py --datasetName MMPbasal_all --discreteTag  --npyDir ../npyImputeG2F_KMeans/
python3 -W ignore main.py --datasetName MMPbasal_all_LTMG  --npyDir ../npyImputeG2F_KMeans/

python3 -W ignore main.py --datasetName MMPbasal_allcell  --npyDir ../npyImputeG2F_KMeans/
python3 -W ignore main.py --datasetName MMPbasal_allcell --discreteTag  --npyDir ../npyImputeG2F_KMeans/

for i in {0..2}
do
    python3 -W ignore main.py --datasetName MMPbasal --reconstr $i --npyDir ../npyImputeG2F_KMeans/
    python3 -W ignore main.py --datasetName MMPbasal --discreteTag --reconstr $i --npyDir ../npyImputeG2F_KMeans/
    python3 -W ignore main.py --datasetName MMPbasal_LTMG --reconstr $i --npyDir ../npyImputeG2F_KMeans/

    python3 -W ignore main.py --datasetName MMPbasal_all --reconstr $i --npyDir ../npyImputeG2F_KMeans/
    python3 -W ignore main.py --datasetName MMPbasal_all --discreteTag --reconstr $i --npyDir ../npyImputeG2F_KMeans/
    python3 -W ignore main.py --datasetName MMPbasal_all_LTMG --reconstr $i --npyDir ../npyImputeG2F_KMeans/

    python3 -W ignore main.py --datasetName MMPbasal_allcell --reconstr $i --npyDir ../npyImputeG2F_KMeans/
    python3 -W ignore main.py --datasetName MMPbasal_allcell --discreteTag --reconstr $i --npyDir ../npyImputeG2F_KMeans/
done