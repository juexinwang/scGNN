for i in {0..8}
do
for j in {11..11}
do
python results_Reading_23dropout.py --methodName $i --splitMode --batchStr $j > run_Results_Impute_$i-$j.sh
done
done

# submit
for i in {0..8}
do
for j in {11..11}
do
sbatch run_Results_Impute_$i-$j.sh
sleep 1
done
done