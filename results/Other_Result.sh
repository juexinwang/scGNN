for i in {0..0}
do
python results_Reading.py --methodName $i --runMode > OtherResults_$i.txt
done

echo 'Start Impute:\n'
for i in {0..0}
do
python results_Reading.py --methodName $i --imputeMode --runMode > OtherResults_Impute_$i.txt
done
