mkdir npyG2E_LK5
mkdir npyG2F_LK5
mkdir npyN2E_LK5
mkdir npyG1E_LK5

for i in {9..13}
do
sbatch run_experimentImpute_2_g_e_LK_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_f_LK_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_e_LK_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_e_LK_$i.sh
sleep 1

done
