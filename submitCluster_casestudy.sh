mkdir casenpyG2E_LK__
mkdir casenpyG2F_LK__


# for i in {3..6}
for i in {4..6}
do
for j in {1..1}
do
for k in {1..4}
do
sbatch run_case_2geLK_$i\_$j\_$k.sh
sleep 1
# sbatch run_case_2reLK_$i\_$j\_$k.sh
# sleep 1
# sbatch run_case_2geLB_$i\_$j\_$k.sh
# sleep 1
# sbatch run_case_2reLB_$i\_$j\_$k.sh
# sleep 1
done
done
done
