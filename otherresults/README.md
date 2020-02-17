Comparing with other methods:
e.g. MAGIC
1. Install MAGIC <https://github.com/KrishnaswamyLab/MAGIC>
2. Make sure the input files are ready: in the root directory of scGNN: (benchPreprocessData.zip can be downloaded from box)
    cd data/sc/
    unzip scData.zip
3. Make sure the label files are ready: change directory in Other_results_impute.py accordingly (AllBench.zip can be downloaded from box)
    cd /home/wangjue/biodata/scData/ 
    unzip allBench.zip
4. Run analysis
    cd otherresults
    bash MAGIC_analysis.sh
5. Change Other_results_Reading.py, check methods used. --runMode means running on the local machine, otherwise generate script for HPC
    bash Other_Results_Evaluation.sh
6. Check results

For all the other methods. don't forget add log(x+1)