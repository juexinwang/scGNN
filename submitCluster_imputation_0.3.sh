mkdir npyImputeG2E_LK_1
mkdir npyImputeG2E_LK_2
mkdir npyImputeG2E_LK_3

for i in {1..3}
do
sbatch run_experimentImpute_2_g_e_LK_$i\_12.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.3.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.6.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.8.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.9.sh
done