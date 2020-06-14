mkdir npy_LK_1
mkdir npy_LK_2
mkdir npy_LK_3

for i in {1..25}
do
sbatch run_experiment_2_g_e_LK_1_$i.sh
sleep 1
sbatch run_experiment_2_g_e_LK_2_$i.sh
sleep 1
sbatch run_experiment_2_g_e_LK_3_$i.sh
sleep 1
done
