
for i in {9,11,12,13}
do
for j in {0..4}
do
sbatch run_experimentImpute_1_g_e_LK_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_g_f_LK_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_n_e_LK_$i\_$j.sh
sleep 1

sbatch run_experimentImpute_1_g_e_LK2_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_g_f_LK2_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_n_e_LK2_$i\_$j.sh
sleep 1

sbatch run_experimentImpute_1_g_e_LK3_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_g_f_LK3_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_n_e_LK3_$i\_$j.sh
sleep 1
done
done

