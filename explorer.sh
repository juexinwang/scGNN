mkdir -p UaeC/npyImputeG1E_LK_1
mkdir -p UaeC/npyImputeG1F_LK_1
mkdir -p UaeC/npyImputeN1E_LK_1

mkdir -p UaeC/npyImputeG1E_LK_2
mkdir -p UaeC/npyImputeG1F_LK_2
mkdir -p UaeC/npyImputeN1E_LK_2

mkdir -p UaeC/npyImputeG1E_LK_3
mkdir -p UaeC/npyImputeG1F_LK_3
mkdir -p UaeC/npyImputeN1E_LK_3

python generatingMethodsBatchshell_graph.py --imputeMode --unweighted
bash submitCluster_Impute_graph.sh
