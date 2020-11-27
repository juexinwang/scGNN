# for i in {0..59}
# do
# for j in {9,11,12,13}
# do
# python results_Reading_recheck.py --methodName $i --splitMode --batchStr $j > run_Results_Impute_$i-$j.sh
# done
# done

# submit
for j in {9,11,12,13}
do
for i in {0..59}
do
sbatch run_Results_Impute_$i-$j.sh
sleep 1
done
done