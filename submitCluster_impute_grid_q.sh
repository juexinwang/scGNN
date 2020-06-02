# mkdir npyImputeG2E_LK_1a
# mkdir npyImputeG2F_LK_1a
# mkdir npyImputeN2E_LK_1a
# mkdir npyImputeG1E_LK_1a

# mkdir npyImputeG2E_LK_2a
# mkdir npyImputeG2F_LK_2a
# mkdir npyImputeN2E_LK_2a
# mkdir npyImputeG1E_LK_2a

# mkdir npyImputeG2E_LK_3a
# mkdir npyImputeG2F_LK_3a
# mkdir npyImputeN2E_LK_3a
# mkdir npyImputeG1E_LK_3a
# for i in {9,11,12,13}
for i in {13..13}
do
for j in {0..27}
do
sbatch run_experimentImpute_2_g_e_qLK1_$i\_$j.sh
sleep 1
# sbatch run_experimentImpute_1_g_e_qLK1_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_2_g_f_qLK1_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_2_n_e_qLK1_$i\_$j.sh
# sleep 1

# sbatch run_experimentImpute_2_g_e_qLK2_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_1_g_e_qLK2_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_2_g_f_qLK2_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_2_n_e_qLK2_$i\_$j.sh
# sleep 1

# sbatch run_experimentImpute_2_g_e_qLK3_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_1_g_e_qLK3_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_2_g_f_qLK3_$i\_$j.sh
# sleep 1
# sbatch run_experimentImpute_2_n_e_LK3_$i\_$j.sh
# sleep 1
done
done

