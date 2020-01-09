python3 -W ignore main.py --datasetName MMPbasal --regulized-type noregu  --npyDir ../npyImputeN2E/
python3 -W ignore main.py --datasetName MMPbasal --discreteTag  --regulized-type noregu  --npyDir ../npyImputeN2E/
python3 -W ignore main.py --datasetName MMPbasal_LTMG --regulized-type noregu  --npyDir ../npyImputeN2E/

python3 -W ignore main.py --datasetName MMPbasal_all --regulized-type noregu  --npyDir ../npyImputeN2E/
python3 -W ignore main.py --datasetName MMPbasal_all --discreteTag  --regulized-type noregu  --npyDir ../npyImputeN2E/
python3 -W ignore main.py --datasetName MMPbasal_all_LTMG --regulized-type noregu  --npyDir ../npyImputeN2E/

python3 -W ignore main.py --datasetName MMPbasal_allcell --regulized-type noregu  --npyDir ../npyImputeN2E/
python3 -W ignore main.py --datasetName MMPbasal_allcell --discreteTag  --regulized-type noregu  --npyDir ../npyImputeN2E/

for i in {0..2}
do
    python3 -W ignore main.py --datasetName MMPbasal --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/
    python3 -W ignore main.py --datasetName MMPbasal --discreteTag  --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/
    python3 -W ignore main.py --datasetName MMPbasal_LTMG --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/

    python3 -W ignore main.py --datasetName MMPbasal_all --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/
    python3 -W ignore main.py --datasetName MMPbasal_all --discreteTag  --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/
    python3 -W ignore main.py --datasetName MMPbasal_all_LTMG --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/

    python3 -W ignore main.py --datasetName MMPbasal_allcell --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/
    python3 -W ignore main.py --datasetName MMPbasal_allcell --discreteTag  --regulized-type noregu --reconstr $i --npyDir ../npyImputeN2E/
done