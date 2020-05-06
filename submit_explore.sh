mkdir npyeG2E
mkdir npyeRG2E

for i in {12..13}
do
sbatch run_explore_$i\_Graph1.sh
sleep 1
sbatch run_explore_$i\_Graph2.sh
sleep 1
sbatch run_explore_$i\_R1.sh
sleep 1
sbatch run_explore_$i\_R2.sh
sleep 1
done