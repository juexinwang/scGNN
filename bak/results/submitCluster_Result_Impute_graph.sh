for i in {0..8}
do
for j in {8,10,11,12}
do
for k in {0.0,0.1,0.9,1.0}
do
python results_Reading_graph.py --methodName $i --splitMode --batchStr $j --regupara $k --imputeMode > run_Results_Impute_$i-$j-$k.sh
done
done
done

# submit
# for i in {0..8}
# do
# for j in {8,10,11,12}
# do
# for k in {0.0,0.1,0.9,1.0}
# do
# sbatch run_Results_Impute_$i-$j-$k.sh
# sleep 1
# done
# done
# done

