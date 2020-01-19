for i in {0..0}
do
python Other_results_Reading.py --methodName $i --runMode > OtherResults_$i.txt
done

echo 'Start Impute:\n'
for i in {0..1}
do
# python Other_results_Reading.py --methodName $i --imputeMode --runMode > OtherResults_Impute_$i.txt
python Other_results_Reading.py --methodName $i --imputeMode > runResults_Impute_$i.sh
done
