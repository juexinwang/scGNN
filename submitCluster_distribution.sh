#submit plotting

for i in {0.1,0.3,0.6,0.8}
do
sbatch plot_G2E_$i\_12.sh

sbatch plot_G2E_$i\_13.sh
done

for i in {0.1,0.3,0.6,0.8}
do
sbatch plot_G2EL_$i\_12.sh
sbatch plot_G1E_$i\_12.sh
sbatch plot_G2F_$i\_12.sh
sbatch plot_N2E_$i\_12.sh

sbatch plot_G2EL_$i\_13.sh
sbatch plot_G1E_$i\_13.sh
sbatch plot_G2F_$i\_13.sh
sbatch plot_N2E_$i\_13.sh
done