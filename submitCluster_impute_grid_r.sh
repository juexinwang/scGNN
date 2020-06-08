mkdir npyImputeG2E_LK_1r
mkdir npyImputeG2F_LK_1r
mkdir npyImputeN2E_LK_1r
mkdir npyImputeG1E_LK_1r

mkdir npyImputeG2E_LK_2r
mkdir npyImputeG2F_LK_2r
mkdir npyImputeN2E_LK_2r
mkdir npyImputeG1E_LK_2r

mkdir npyImputeG2E_LK_3r
mkdir npyImputeG2F_LK_3r
mkdir npyImputeN2E_LK_3r
mkdir npyImputeG1E_LK_3r
for i in {9,11,12,13}
# for i in {12,13}
do
for j in {5..5}
do
sbatch run_experimentImpute_2_g_e_LK1_$i\_$j.sh
sbatch run_experimentImpute_1_g_e_LK1_$i\_$j.sh
sbatch run_experimentImpute_2_g_f_LK1_$i\_$j.sh
sbatch run_experimentImpute_2_n_e_LK1_$i\_$j.sh

sbatch run_experimentImpute_2_g_e_LK2_$i\_$j.sh
sbatch run_experimentImpute_1_g_e_LK2_$i\_$j.sh
sbatch run_experimentImpute_2_g_f_LK2_$i\_$j.sh
sbatch run_experimentImpute_2_n_e_LK2_$i\_$j.sh

sbatch run_experimentImpute_2_g_e_LK3_$i\_$j.sh
sbatch run_experimentImpute_1_g_e_LK3_$i\_$j.sh
sbatch run_experimentImpute_2_g_f_LK3_$i\_$j.sh
sbatch run_experimentImpute_2_n_e_LK3_$i\_$j.sh
done
done

