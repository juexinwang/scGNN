module load miniconda3
source activate conda_R
#dir='UaeC'
#dir='UaeO'
#dir='WaeC'
#dir='WaeO'
dir=''
for dataset in {'9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel'}
# for dataset in {'13.Zeisel','13.Zeisel'}
do
echo $dataset
tdataset=$(echo $dataset | cut -d'.' -f 2)
for ii in {'0.0','0.1','0.3','0.9'}
do
for jj in {'0.0-0.9','0.1-0.0','0.1-0.1','0.1-0.3','0.1-0.9','0.3-0.1','0.9-0.1'} 
do
para="$ii-$jj"
for j in {1..3}
do
python -W ignore results_impute_graph.py --datasetName $dataset --regulized-type LTMG  --benchmark --labelFilename /home/jwang/data/scData/$dataset\/$tdataset\_cell_label.csv --n-clusters 4  --regupara $para --npyDir ../$dir\/npyImputeG2E_LK_$j\/
done
done
done
done
