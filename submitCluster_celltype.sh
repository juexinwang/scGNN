mkdir npyG2E_LK_1
mkdir npyG2F_LK_1
mkdir npyN2E_LK_1
mkdir npyG1E_LK_1

mkdir npyG2E_LK_2
mkdir npyG2F_LK_2
mkdir npyN2E_LK_2
mkdir npyG1E_LK_2

mkdir npyG2E_LK_3
mkdir npyG2F_LK_3
mkdir npyN2E_LK_3
mkdir npyG1E_LK_3

for i in {9,11,12,13}
do
sbatch run_experiment_2_g_e_LK_1_$i.sh
sleep 1
sbatch run_experiment_2_g_f_LK_1_$i.sh
sleep 1
sbatch run_experiment_2_n_e_LK_1_$i.sh
sleep 1
sbatch run_experiment_1_g_e_LK_1_$i.sh
sleep 1

sbatch run_experiment_2_g_e_LK_2_$i.sh
sleep 1
sbatch run_experiment_2_g_f_LK_2_$i.sh
sleep 1
sbatch run_experiment_2_n_e_LK_2_$i.sh
sleep 1
sbatch run_experiment_1_g_e_LK_2_$i.sh
sleep 1

sbatch run_experiment_2_g_e_LK_3_$i.sh
sleep 1
sbatch run_experiment_2_g_f_LK_3_$i.sh
sleep 1
sbatch run_experiment_2_n_e_LK_3_$i.sh
sleep 1
sbatch run_experiment_1_g_e_LK_3_$i.sh
sleep 1

done
