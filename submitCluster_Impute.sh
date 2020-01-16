mkdir npyImputeG1E
mkdir npyImputeG1F
mkdir npyImputeN1E
mkdir npyImputeN1F
mkdir npyImputeG2E_AffinityPropagation
mkdir npyImputeG2E_AgglomerativeClustering
mkdir npyImputeG2E_Birch
mkdir npyImputeG2E_KMeans
mkdir npyImputeG2E_SpectralClustering
mkdir npyImputeG2E
mkdir npyImputeG2F_AffinityPropagation
mkdir npyImputeG2F_AgglomerativeClustering
mkdir npyImputeG2F_Birch
mkdir npyImputeG2F_KMeans
mkdir npyImputeG2F_SpectralClustering
mkdir npyImputeG2F
mkdir npyImputeN2E_AffinityPropagation
mkdir npyImputeN2E_AgglomerativeClustering
mkdir npyImputeN2E_Birch
mkdir npyImputeN2E_KMeans
mkdir npyImputeN2E_SpectralClustering
mkdir npyImputeN2E
mkdir npyImputeN2F_AffinityPropagation
mkdir npyImputeN2F_AgglomerativeClustering
mkdir npyImputeN2F_Birch
mkdir npyImputeN2F_KMeans
mkdir npyImputeN2F_SpectralClustering
mkdir npyImputeN2F

for i in {12..25}
do
sbatch run_experimentImpute_1_g_e_$i.sh
sleep 3
done

for i in {12..25}
do
sbatch run_experimentImpute_1_n_e_$i.sh
sleep 3
done

for i in {12..25}
do
sbatch run_experimentImpute_1_n_f_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_e_AffinityPropagation_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_e_AgglomerativeClustering_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_e_Birch_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_e_KMeans_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_e_SpectralClustering_$i.sh
sleep 3
done

# for i in {9..25}
# do
# sbatch run_experimentImpute_2_n_e_$i.sh
# sleep 3
# done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_f_AffinityPropagation_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_f_AgglomerativeClustering_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_f_Birch_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_f_KMeans_$i.sh
sleep 3
done

for i in {9..25}
do
sbatch run_experimentImpute_2_n_f_SpectralClustering_$i.sh
sleep 3
done

# for i in {9..25}
# do
# sbatch run_experimentImpute_2_n_f_$i.sh
# sleep 3
# done

