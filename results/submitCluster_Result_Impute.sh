# for i in {0..62}
# do
# python results_Reading.py --methodName $i --imputeMode > run_Results_Impute_$i.sh
# done

# # submit
# for i in {0..62}
# do
# sbatch run_Results_Impute_$i.sh
# sleep 1
# done

## split mode
for i in {0..62}
do
for j in {0..14}
do
python results_Reading.py --methodName $i --splitMode --batchStr $j --imputeMode > run_Results_Impute_$i-$j.sh
done
done

# submit
# for i in {0..62}
# do
# for j in {0..12}
# do
# sbatch run_Results_Impute_$i-$j.sh
# sleep 1
# done
# done