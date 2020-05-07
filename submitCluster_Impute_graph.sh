mkdir npyImputeG1E_LK
mkdir npyImputeG1F_LK
mkdir npyImputeN1E_LK

mkdir npyImputeG1E_LK_2
mkdir npyImputeG1F_LK_2
mkdir npyImputeN1E_LK_2

mkdir npyImputeG1E_LK_3
mkdir npyImputeG1F_LK_3
mkdir npyImputeN1E_LK_3

for i in {9,11,12,13}
do
for j in {0..3}
do
sbatch run_experimentImpute_1_n_e_LK_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_g_f_LK_$i\_$j.sh
sleep 1
sbatch run_experimentImpute_1_n_e_LK_$i\_$j.sh
sleep 1

# sbatch run_experimentImpute_1_n_e_LK2_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_1_g_f_LK2_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_1_n_e_LK2_$i\_$j.sh
# sleep 1

# sbatch run_experimentImpute_1_n_e_LK3_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_1_g_f_LK3_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_1_n_e_LK3_$i\_$j.sh
# sleep 1
done
done

