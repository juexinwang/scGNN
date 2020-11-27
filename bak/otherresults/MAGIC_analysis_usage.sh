module load miniconda3
source activate conda_R
for i in {0.1,0.3,0.6,0.8}
do
for j in {1..10}
do
python -W ignore MAGIC_impute_usage.py --datasetName 12.Klein --ratio $i --replica $j
done
done