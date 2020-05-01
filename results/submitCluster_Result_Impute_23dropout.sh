for i in {0..9}
do
for j in {12..12}
do
python results_Reading_23dropout.py --methodName $i --splitMode --batchStr $j > run_Results_Impute_$i-$j.sh
done
done

# submit
for i in {0..9}
do
for j in {12..12}
do
sbatch run_Results_Impute_$i-$j.sh
sleep 1
done
done