python3 -W ignore results_impute.py --datasetName MMPbasal  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal --discreteTag  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal_LTMG  --npyDir ../npyImputeG2E_Birch/

python3 -W ignore results_impute.py --datasetName MMPbasal_all  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal_all --discreteTag  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal_all_LTMG  --npyDir ../npyImputeG2E_Birch/

python3 -W ignore results_impute.py --datasetName MMPbasal_allcell  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --discreteTag  --npyDir ../npyImputeG2E_Birch/

python3 -W ignore results_impute.py --datasetName MMPbasal_2000  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal_2000 --discreteTag  --npyDir ../npyImputeG2E_Birch/
python3 -W ignore results_impute.py --datasetName MMPbasal_2000_LTMG  --npyDir ../npyImputeG2E_Birch/

for i in {0..4}
do
    python3 -W ignore results_impute.py --datasetName MMPbasal --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal --discreteTag --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal_LTMG --reconstr $i --npyDir ../npyImputeG2E_Birch/

    python3 -W ignore results_impute.py --datasetName MMPbasal_all --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal_all --discreteTag --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal_all_LTMG --reconstr $i --npyDir ../npyImputeG2E_Birch/

    python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --discreteTag --reconstr $i --npyDir ../npyImputeG2E_Birch/

    python3 -W ignore results_impute.py --datasetName MMPbasal_2000 --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal_2000 --discreteTag --reconstr $i --npyDir ../npyImputeG2E_Birch/
    python3 -W ignore results_impute.py --datasetName MMPbasal_2000_LTMG --reconstr $i --npyDir ../npyImputeG2E_Birch/
done