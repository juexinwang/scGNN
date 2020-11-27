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

## complex
## split mode
# for i in {0..62}
# do
# for j in {0..12}
# do
# python results_Reading.py --methodName $i --splitMode --batchStr $j --imputeMode > run_Results_Impute_$i-$j.sh
# done
# done

# # submit
# for i in {0..62}
# do
# for j in {0..12}
# do
# sbatch run_Results_Impute_$i-$j.sh
# sleep 1
# done
# done

## selected
## split mode
# for i in {0..19}
# for i in {0..15}
for i in {0..53}
do
for j in {8..12}
do
python results_Reading.py --methodName $i --splitMode --batchStr $j --imputeMode > run_Results_Impute_$i-$j.sh
done
done

# submit
# for i in {0..19}
# for i in {0..15}
for i in {0..53}
do
for j in {8..12}
do
sbatch run_Results_Impute_$i-$j.sh
sleep 1
done
done