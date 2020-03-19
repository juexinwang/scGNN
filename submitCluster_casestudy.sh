mkdir casenpyG2E_LK
mkdir casenpyR2E_LK
mkdir casenpyG2E_LB
mkdir casenpyR2E_LB

for i in {1..1}
do
for j in {1..3}
do
for k in {1..4}
do
sbatch run_case_2geLK_$i\_$j\_$k.sh
sleep 1
sbatch run_case_2reLK_$i\_$j\_$k.sh
sleep 1
sbatch run_case_2geLB_$i\_$j\_$k.sh
sleep 1
sbatch run_case_2reLB_$i\_$j\_$k.sh
sleep 1
done
done
done
