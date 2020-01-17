for i in {0..27}
do
sbatch run_Results_Impute_$i.sh
sleep 3
done