mkdir npyImputeG1B
mkdir npyImputeG1E
mkdir npyImputeG1F
mkdir npyImputeR1B
mkdir npyImputeR1E
mkdir npyImputeR1F
mkdir npyImputeN1B
mkdir npyImputeN1E
mkdir npyImputeN1F
mkdir npyImputeG2B
mkdir npyImputeG2E
mkdir npyImputeG2F
mkdir npyImputeR2B
mkdir npyImputeR2E
mkdir npyImputeR2F
mkdir npyImputeN2B
mkdir npyImputeN2E
mkdir npyImputeN2F

mkdir npyImputeG1B_LK
mkdir npyImputeG1E_LK
mkdir npyImputeG1F_LK
mkdir npyImputeR1B_LK
mkdir npyImputeR1E_LK
mkdir npyImputeR1F_LK
mkdir npyImputeN1B_LK
mkdir npyImputeN1E_LK
mkdir npyImputeN1F_LK
mkdir npyImputeG2B_LK
mkdir npyImputeG2E_LK
mkdir npyImputeG2F_LK
mkdir npyImputeR2B_LK
mkdir npyImputeR2E_LK
mkdir npyImputeR2F_LK
mkdir npyImputeN2B_LK
mkdir npyImputeN2E_LK
mkdir npyImputeN2F_LK

mkdir npyImputeG1B_LB
mkdir npyImputeG1E_LB
mkdir npyImputeG1F_LB
mkdir npyImputeR1B_LB
mkdir npyImputeR1E_LB
mkdir npyImputeR1F_LB
mkdir npyImputeN1B_LB
mkdir npyImputeN1E_LB
mkdir npyImputeN1F_LB
mkdir npyImputeG2B_LB
mkdir npyImputeG2E_LB
mkdir npyImputeG2F_LB
mkdir npyImputeR2B_LB
mkdir npyImputeR2E_LB
mkdir npyImputeR2F_LB
mkdir npyImputeN2B_LB
mkdir npyImputeN2E_LB
mkdir npyImputeN2F_LB

for i in {1..13}
do
# sbatch run_experimentImpute_1_g_b_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_g_e_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_g_f_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_r_b_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_r_e_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_r_f_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_n_b_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_n_e_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_n_f_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_g_b_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_g_e_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_g_f_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_r_b_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_r_e_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_r_f_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_n_b_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_n_e_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_n_f_$i.sh
# sleep 1

# sbatch run_experimentImpute_1_g_b_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_g_e_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_g_f_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_r_b_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_r_e_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_r_f_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_n_b_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_n_e_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_1_n_f_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_g_b_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_g_e_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_g_f_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_r_b_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_r_e_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_r_f_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_n_b_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_n_e_LK_$i.sh
# sleep 1
# sbatch run_experimentImpute_2_n_f_LK_$i.sh
# sleep 1

sbatch run_experimentImpute_1_g_b_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_g_f_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_r_b_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_r_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_r_f_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_n_b_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_n_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_1_n_f_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_b_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_f_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_b_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_f_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_b_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_n_f_LB_$i.sh
sleep 1
done
