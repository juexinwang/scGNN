mkdir casenpyG2E_LK_
mkdir casenpyR2E_LK_
mkdir casenpyG2E_LB_
mkdir casenpyR2E_LB_

for i in {1..7}
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
