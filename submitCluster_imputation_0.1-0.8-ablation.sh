mkdir npyImputeG2E_1
mkdir npyImputeG2EL_1
mkdir npyImputeG2F_1
mkdir npyImputeN2E_1
mkdir npyImputeG1E_1

mkdir npyImputeG2E_2
mkdir npyImputeG2EL_2
mkdir npyImputeG2F_2
mkdir npyImputeN2E_2
mkdir npyImputeG1E_2

mkdir npyImputeG2E_3
mkdir npyImputeG2EL_3
mkdir npyImputeG2F_3
mkdir npyImputeN2E_3
mkdir npyImputeG1E_3

for i in {1..3}
do
for j in {0.1,0.3,0.6,0.8}
do
sbatch run_experimentImpute_1_g_e_$i\_12_$j\.sh
sbatch run_experimentImpute_2_g_e_$i\_12_$j\.sh
sbatch run_experimentImpute_2_g_e_L_$i\_12_$j\.sh
sbatch run_experimentImpute_2_g_f_$i\_12_$j\.sh
sbatch run_experimentImpute_2_n_e_$i\_12_$j\.sh

sbatch run_experimentImpute_1_g_e_$i\_13_$j\.sh
sbatch run_experimentImpute_2_g_e_$i\_13_$j\.sh
sbatch run_experimentImpute_2_g_e_L_$i\_13_$j\.sh
sbatch run_experimentImpute_2_g_f_$i\_13_$j\.sh
sbatch run_experimentImpute_2_n_e_$i\_13_$j\.sh
done
done