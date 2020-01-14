python -W ignore results_impute.py --datasetName MMPbasal --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal --discreteTag --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal_LTMG --regulized-type noregu  --npyDir ../npyImputeN1F/

python -W ignore results_impute.py --datasetName MMPbasal_all --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal_all --discreteTag --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal_all_LTMG --regulized-type noregu  --npyDir ../npyImputeN1F/

python -W ignore results_impute.py --datasetName MMPbasal_allcell --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal_allcell --discreteTag --regulized-type noregu  --npyDir ../npyImputeN1F/

python -W ignore results_impute.py --datasetName MMPbasal_2000 --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal_2000 --discreteTag --regulized-type noregu  --npyDir ../npyImputeN1F/
python -W ignore results_impute.py --datasetName MMPbasal_2000_LTMG --regulized-type noregu  --npyDir ../npyImputeN1F/

for i in {0..4}
do
    python -W ignore results_impute.py --datasetName MMPbasal --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal --discreteTag --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal_LTMG --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/

    python -W ignore results_impute.py --datasetName MMPbasal_all --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal_all --discreteTag --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal_all_LTMG --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/

    python -W ignore results_impute.py --datasetName MMPbasal_allcell --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal_allcell --discreteTag --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/

    python -W ignore results_impute.py --datasetName MMPbasal_2000 --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal_2000 --discreteTag --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
    python -W ignore results_impute.py --datasetName MMPbasal_2000_LTMG --regulized-type noregu --reconstr $i --npyDir ../npyImputeN1F/
done