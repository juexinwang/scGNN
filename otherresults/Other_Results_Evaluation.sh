# for i in {0..0}
# do
# python Other_results_Reading.py --methodName $i > runOtherResults_$i.sh
# done

echo 'Start Impute:\n'
for i in {0..0}
do
python Other_results_Reading.py --methodName $i --imputeMode --runMode > OtherResults_Impute_$i.txt
# python Other_results_Reading.py --methodName $i --imputeMode > runOtherResults_Impute_$i.sh
done
