for i in {9..25}
do
sbatch run_experimentImpute_2_n_e_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_f_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_g_e_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_g_f_$i.sh
sleep 3
done

