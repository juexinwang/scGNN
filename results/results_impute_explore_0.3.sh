module load miniconda3
source activate conda_R
for ii in {'0.1','0.3','0.6','0.8','0.9'}
do
for j in {1..3}
do
python -W ignore results_impute_graph.py --datasetName 12.Klein --regulized-type LTMG  --benchmark --labelFilename /home/jwang/data/scData/12.Klein/Klein_cell_label.csv --regupara $para --ratio $ii --npyDir ../npyImputeG1E_LK_$j\/
done
done
