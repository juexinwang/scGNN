for i in {0..62}
do
python results_Reading.py --methodName $i --imputeMode > run_Results_Impute_$i.sh
done

submit
for i in {0..62}
do
sbatch run_Results_Impute_$i.sh
sleep 1
done