for i in {15..25}
do
sbatch run_experimentImpute_1_g_e_$i.sh
sleep 3
done

for i in {15..25}
do
sbatch run_experimentImpute_1_n_e_$i.sh
sleep 3
done

for i in {15..25}
do
sbatch run_experimentImpute_1_n_f_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_e_AffinityPropagation_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_e_AgglomerativeClustering_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_e_Birch_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_e_KMeans_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_e_SpectralClustering_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_e_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_f_AffinityPropagation_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_f_AgglomerativeClustering_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_f_Birch_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_f_KMeans_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_f_SpectralClustering_$i.sh
sleep 3
done

for i in {9,10,11,15,16,17,18,19,20,21,22,23,24,25}
do
sbatch run_experimentImpute_2_n_f_$i.sh
sleep 3
done

