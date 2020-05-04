mkdir npyImputeG1B
mkdir npyImputeG1E
mkdir npyImputeR1B
mkdir npyImputeR1E
mkdir npyImputeG2B
mkdir npyImputeG2E
mkdir npyImputeR2B
mkdir npyImputeR2E
mkdir npyImputeG2B_Birch
# mkdir npyImputeG2B_BirchN
mkdir npyImputeG2B_KMeans
mkdir npyImputeG2E_Birch
# mkdir npyImputeG2E_BirchN
mkdir npyImputeG2E_KMeans
mkdir npyImputeR2B_Birch
# mkdir npyImputeR2B_BirchN
mkdir npyImputeR2B_KMeans
mkdir npyImputeR2E_Birch
# mkdir npyImputeR2E_BirchN
mkdir npyImputeR2E_KMeans


for i in {1..13}
do
sbatch run_experimentImpute_1_g_b_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_e_$i.sh
sleep 1
sbatch run_experimentImpute_1_r_b_$i.sh
sleep 1
sbatch run_experimentImpute_1_r_e_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_b_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_e_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_b_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_e_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_b_Birch_$i.sh
sleep 1
# sbatch run_experimentImpute_2_g_b_BirchN_$i.sh
# sleep 1
sbatch run_experimentImpute_2_g_b_KMeans_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_e_Birch_$i.sh
sleep 1
# sbatch run_experimentImpute_2_g_e_BirchN_$i.sh
# sleep 1
sbatch run_experimentImpute_2_g_e_KMeans_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_b_Birch_$i.sh
sleep 1
# sbatch run_experimentImpute_2_r_b_BirchN_$i.sh
# sleep 1
sbatch run_experimentImpute_2_r_b_KMeans_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_e_Birch_$i.sh
sleep 1
# sbatch run_experimentImpute_2_r_e_BirchN_$i.sh
# sleep 1
sbatch run_experimentImpute_2_r_e_KMeans_$i.sh
sleep 1
done