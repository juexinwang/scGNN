mkdir -p UaeC/npyImputeG1E_LK_1
mkdir -p UaeC/npyImputeG1F_LK_1
mkdir -p UaeC/npyImputeN1E_LK_1

mkdir -p UaeC/npyImputeG1E_LK_2
mkdir -p UaeC/npyImputeG1F_LK_2
mkdir -p UaeC/npyImputeN1E_LK_2

mkdir -p UaeC/npyImputeG1E_LK_3
mkdir -p UaeC/npyImputeG1F_LK_3
mkdir -p UaeC/npyImputeN1E_LK_3

python generatingMethodsBatchshell_graph.py --imputeMode --adjtype unweighted
bash submitCluster_Impute_graph.sh

mkdir -p UaeO/npyImputeG1E_LK_1
mkdir -p UaeO/npyImputeG1F_LK_1
mkdir -p UaeO/npyImputeN1E_LK_1

mkdir -p UaeO/npyImputeG1E_LK_2
mkdir -p UaeO/npyImputeG1F_LK_2
mkdir -p UaeO/npyImputeN1E_LK_2

mkdir -p UaeO/npyImputeG1E_LK_3
mkdir -p UaeO/npyImputeG1F_LK_3
mkdir -p UaeO/npyImputeN1E_LK_3

python generatingMethodsBatchshell_graph.py --imputeMode --adjtype unweighted --aeOriginal
bash submitCluster_Impute_graph.sh

mkdir -p WaeC/npyImputeG1E_LK_1
mkdir -p WaeC/npyImputeG1F_LK_1
mkdir -p WaeC/npyImputeN1E_LK_1

mkdir -p WaeC/npyImputeG1E_LK_2
mkdir -p WaeC/npyImputeG1F_LK_2
mkdir -p WaeC/npyImputeN1E_LK_2

mkdir -p WaeC/npyImputeG1E_LK_3
mkdir -p WaeC/npyImputeG1F_LK_3
mkdir -p WaeC/npyImputeN1E_LK_3

python generatingMethodsBatchshell_graph.py --imputeMode --adjtype weighted
bash submitCluster_Impute_graph.sh

mkdir -p WaeO/npyImputeG1E_LK_1
mkdir -p WaeO/npyImputeG1F_LK_1
mkdir -p WaeO/npyImputeN1E_LK_1

mkdir -p WaeO/npyImputeG1E_LK_2
mkdir -p WaeO/npyImputeG1F_LK_2
mkdir -p WaeO/npyImputeN1E_LK_2

mkdir -p WaeO/npyImputeG1E_LK_3
mkdir -p WaeO/npyImputeG1F_LK_3
mkdir -p WaeO/npyImputeN1E_LK_3

python generatingMethodsBatchshell_graph.py --imputeMode --adjtype weighted --aeOriginal
bash submitCluster_Impute_graph.sh
