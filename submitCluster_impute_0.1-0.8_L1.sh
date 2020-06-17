mkdir npyImputeG2E_1
mkdir npyImputeG2E_2
mkdir npyImputeG2E_3
mkdir npyImputeG2E_4
mkdir npyImputeG2E_5
mkdir npyImputeG2E_6
mkdir npyImputeG2E_7
mkdir npyImputeG2E_8
mkdir npyImputeG2E_9

for i in {1..9}
do
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.1.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.2.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.3.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.4.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.5.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.6.sh
# sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.7.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_12_0.8.sh

sbatch run_experimentImpute_2_g_e_LK_$i\_13_0.1.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_13_0.3.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_13_0.6.sh
sbatch run_experimentImpute_2_g_e_LK_$i\_13_0.8.sh
done