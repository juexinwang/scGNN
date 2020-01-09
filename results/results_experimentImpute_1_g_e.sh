python3 -W ignore results_impute.py --datasetName MMPbasal --npyDir ../npyImputeG1E/
python3 -W ignore results_impute.py --datasetName MMPbasal --discreteTag  --npyDir ../npyImputeG1E/
python3 -W ignore results_impute.py --datasetName MMPbasal_LTMG --npyDir ../npyImputeG1E/

python3 -W ignore results_impute.py --datasetName MMPbasal_all --npyDir ../npyImputeG1E/
python3 -W ignore results_impute.py --datasetName MMPbasal_all --discreteTag --npyDir ../npyImputeG1E/
python3 -W ignore results_impute.py --datasetName MMPbasal_all_LTMG --npyDir ../npyImputeG1E/

python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --npyDir ../npyImputeG1E/
python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --discreteTag  --npyDir ../npyImputeG1E/

for i in {0..2}
do
    python3 -W ignore results_impute.py --datasetName MMPbasal --reconstr $i --npyDir ../npyImputeG1E/
    python3 -W ignore results_impute.py --datasetName MMPbasal --discreteTag  --reconstr $i --npyDir ../npyImputeG1E/
    python3 -W ignore results_impute.py --datasetName MMPbasal_LTMG --reconstr $i --npyDir ../npyImputeG1E/

    python3 -W ignore results_impute.py --datasetName MMPbasal_all --reconstr $i --npyDir ../npyImputeG1E/
    python3 -W ignore results_impute.py --datasetName MMPbasal_all --discreteTag --reconstr $i --npyDir ../npyImputeG1E/
    python3 -W ignore results_impute.py --datasetName MMPbasal_all_LTMG --reconstr $i --npyDir ../npyImputeG1E/

    python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --reconstr $i --npyDir ../npyImputeG1E/
    python3 -W ignore results_impute.py --datasetName MMPbasal_allcell --discreteTag  --reconstr $i --npyDir ../npyImputeG1E/
done