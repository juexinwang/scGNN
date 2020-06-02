mkdir npyImputeG2E_LK_1a
mkdir npyImputeG2F_LK_1a
mkdir npyImputeN2E_LK_1a
mkdir npyImputeG1E_LK_1a

mkdir npyImputeG2E_LK_2a
mkdir npyImputeG2F_LK_2a
mkdir npyImputeN2E_LK_2a
mkdir npyImputeG1E_LK_2a

mkdir npyImputeG2E_LK_3a
mkdir npyImputeG2F_LK_3a
mkdir npyImputeN2E_LK_3a
mkdir npyImputeG1E_LK_3a

for i in {12,13}
do
sbatch run_experimentImpute_2_g_e_LK_1_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_f_LK_1_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_e_LK_1_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_e_LK_1_$i.sh
sleep 1

sbatch run_experimentImpute_2_g_e_LK_2_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_f_LK_2_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_e_LK_2_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_e_LK_2_$i.sh
sleep 1

sbatch run_experimentImpute_2_g_e_LK_3_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_f_LK_3_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_e_LK_3_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_e_LK_3_$i.sh
sleep 1
done
