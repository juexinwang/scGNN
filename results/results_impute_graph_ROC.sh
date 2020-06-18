module load miniconda3
source activate conda_R
for ii in {'0.1','0.3','0.6','0.8'}
do
python -W ignore results_impute_graph_ROC.py --datasetName 12.Klein --labelFilename /home/jwang/data/scData/12.Klein/Klein_cell_label.csv --ratio $ii
done

for ii in {'0.1','0.3','0.6','0.8'}
do
python -W ignore results_impute_graph_ROC.py --datasetName 13.Zeisel --labelFilename /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv --ratio $ii
done