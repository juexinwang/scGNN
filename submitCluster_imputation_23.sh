mkdir npyImputeG2E_LK
mkdir npyImputeG2F_LK
mkdir npyImputeN2E_LK
mkdir npyImputeG1E_LK

mkdir npyImputeG2E_LK_2
mkdir npyImputeG2F_LK_2
mkdir npyImputeN2E_LK_2
mkdir npyImputeG1E_LK_2

mkdir npyImputeG2E_LK_3
mkdir npyImputeG2F_LK_3
mkdir npyImputeN2E_LK_3
mkdir npyImputeG1E_LK_3

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
