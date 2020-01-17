for i in {0..27}
do
python results_Reading.py --methodName $i > run_Results_Celltype_$i.sh
done

# submit
for i in {0..27}
do
sbatch run_Results_Celltype_$i.sh
sleep 3
done