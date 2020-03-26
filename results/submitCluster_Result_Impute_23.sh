for i in {0..3}
do
for j in {0..12}
do
python results_Reading_23.py --methodName $i --splitMode --batchStr $j --imputeMode > run_Results_Impute_$i-$j.sh
done
done

# submit
for i in {0..3}
do
for j in {0..12}
do
sbatch run_Results_Impute_$i-$j.sh
sleep 1
done
done