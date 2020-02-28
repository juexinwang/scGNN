# for i in {0..62}
# do
# python results_Reading.py --methodName $i > run_Results_Celltype_$i.sh
# done

# submit
# for i in {0..62}
# do
# sbatch run_Results_Celltype_$i.sh
# sleep 3
# done

#split mode
for i in {0..62}
do
for j in {0..12}
do
python results_Reading.py --methodName $i --splitMode --batchStr $j > run_Results_Celltype_$i-$j.sh
done
done

# submit
for i in {0..62}
do
for j in {0..12}
do
sbatch run_Results_Celltype_$i-$j.sh
sleep 1
done
done