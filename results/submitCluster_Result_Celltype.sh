# for i in {0..27}
# do
# python results_Reading.py --methodName $i > run_Results_Celltype_$i.sh
# done

# submit
# for i in {0..27}
# do
# sbatch run_Results_Celltype_$i.sh
# sleep 3
# done

#split mode
for i in {0..27}
do
for j in {0..1}
do
python results_Reading.py --methodName $i --splitMode --batchStr $j > run_Results_Celltype_$i_$j.sh
done
done

# submit
for i in {0..27}
do
for j in {0..1}
do
sbatch run_Results_Celltype_$i_$j.sh
sleep 3
done
done