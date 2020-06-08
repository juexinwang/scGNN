mkdir npyImputeG2E_LK_1
mkdir npyImputeG2E_LK_2
mkdir npyImputeG2E_LK_3
mkdir npyImputeG2E_LK_4
mkdir npyImputeG2E_LK_5
mkdir npyImputeG2E_LK_6
mkdir npyImputeG2E_LK_7
mkdir npyImputeG2E_LK_8
mkdir npyImputeG2E_LK_9
mkdir npyImputeG2E_LK_10

for i in {1..10}
do
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.1.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.2.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.3.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.4.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.5.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.6.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.7.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.8.sh
done