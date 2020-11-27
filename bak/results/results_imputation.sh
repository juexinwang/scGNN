module load miniconda3
source activate conda_R
for dataset in {'9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel'}
do
echo $dataset
tdataset=$(echo $dataset | cut -d'.' -f 2)
for j in {1..3}
do
python -W ignore results_impute_graph.py --datasetName $dataset --regulized-type LTMG  --benchmark --labelFilename /home/jwang/data/scData/$dataset\/$tdataset\_cell_label.csv --npyDir ../npyImputeG2E_LK_$j\/
done
done