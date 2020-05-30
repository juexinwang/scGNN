module load miniconda3
source activate conda_R
for ii in {'0.1','0.3','0.6','0.8'}
do
for j in {1..10}
do
python -W ignore results_impute_graph.py --datasetName 12.Klein --regulized-type LTMG  --benchmark --labelFilename /home/jwang/data/scData/12.Klein/Klein_cell_label.csv --ratio $ii --npyDir ../npyImputeG2E_LK_$j\/
done
done
