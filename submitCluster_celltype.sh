mkdir npyG2E_LK
mkdir npyG2F_LK
mkdir npyN2E_LK
mkdir npyG1E_LK

for i in {9,11,12,13}
do
sbatch run_experiment_2_g_e_LK_$i.sh
sleep 1
sbatch run_experiment_2_g_f_LK_$i.sh
sleep 1
sbatch run_experiment_2_n_e_LK_$i.sh
sleep 1
sbatch run_experiment_1_g_e_LK_$i.sh
sleep 1

done
