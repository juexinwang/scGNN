mkdir npyG1E
mkdir npyG1F
mkdir npyN1E
mkdir npyN1F
mkdir npyG2E_AffinityPropagation
mkdir npyG2E_AgglomerativeClustering
mkdir npyG2E_Birch
mkdir npyG2E_KMeans
mkdir npyG2E_SpectralClustering
mkdir npyG2E
mkdir npyG2F_AffinityPropagation
mkdir npyG2F_AgglomerativeClustering
mkdir npyG2F_Birch
mkdir npyG2F_KMeans
mkdir npyG2F_SpectralClustering
mkdir npyG2F
mkdir npyN2E_AffinityPropagation
mkdir npyN2E_AgglomerativeClustering
mkdir npyN2E_Birch
mkdir npyN2E_KMeans
mkdir npyN2E_SpectralClustering
mkdir npyN2E
mkdir npyN2F_AffinityPropagation
mkdir npyN2F_AgglomerativeClustering
mkdir npyN2F_Birch
mkdir npyN2F_KMeans
mkdir npyN2F_SpectralClustering
mkdir npyN2F

for i in {9..25}
do
sbatch run_experiment_1_g_e_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_1_n_e_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_1_n_f_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_e_AffinityPropagation_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_e_AgglomerativeClustering_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_e_Birch_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_e_KMeans_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_e_SpectralClustering_$i.sh
sleep 3
done

# for i in {9..25}
# do
# sbatch run_experiment_2_n_e_$i.sh
# sleep 3
# done

for i in {9..25}
do
sbatch run_experiment_2_n_f_AffinityPropagation_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_f_AgglomerativeClustering_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_f_Birch_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_f_KMeans_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experiment_2_n_f_SpectralClustering_$i.sh
sleep 3
done

# for i in {9..25}
# do
# sbatch run_experiment_2_n_f_$i.sh
# sleep 3
# done

