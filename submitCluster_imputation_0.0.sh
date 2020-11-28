for i in {1}
do
for j in {0.0}
do
sbatch run_experimentImpute_2_g_e_$i\_9_$j\.sh

sbatch run_experimentImpute_2_g_e_$i\_11_$j\.sh

sbatch run_experimentImpute_2_g_e_$i\_12_$j\.sh

sbatch run_experimentImpute_2_g_e_$i\_13_$j\.sh

done
done