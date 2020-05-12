module load miniconda3
source activate conda_R
dir='UaeC/'
#dir='UaeO/'
#dir='WaeC/'
#dir='WaeO/'
for dataset in {'9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel'}
do
echo $dataset
tdataset=$(echo $dataset | cut -d'.' -f 2)
for i in {0.0,0.1,0.5,0.9,1.0}
do
for j in {1..3}
do
python -W ignore results_impute_graph.py --datasetName $dataset --regulized-type LTMG  --benchmark --labelFilename /home/jwang/data/scData/$dataset\/$tdataset\_cell_label.csv --n-clusters 4  --reconstr 0  --regupara $i --npyDir ../$dir\/npyImputeG1E_LK_$j\/
done
done
done
