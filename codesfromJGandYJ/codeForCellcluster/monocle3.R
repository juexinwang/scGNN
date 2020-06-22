#install.packages('BiocManager')
#BiocManager::install(version = "3.9")
#install.packages("C:\\Program Files\\R\\R-3.6.0\\library\\monocle3-0.2.0.tar.gz", repos = NULL)
library('monocle3')
library(dplyr)

#expression_matrix <- readRDS(url("http://staff.washington.edu/hpliner/data/cao_l2_expression.rds"))
#cell_metadata <- readRDS(url("http://staff.washington.edu/hpliner/data/cao_l2_colData.rds"))
#gene_annotation <- readRDS(url("http://staff.washington.edu/hpliner/data/cao_l2_rowData.rds"))


expression_matrix <- read.table('E:\\SingleCell\\allBench\\9.Chung\\T2000_UsingOriginalMatrix\\T2000_expression.txt',header = TRUE,row.names=1)

gene <- expression_matrix[1]
colnames(gene)[1] <- 'gene_short_name'

#imputation results
cellfea <- read.csv('E:\\SingleCell\\allBench\\9.Chung\\Chung_cell_label.csv',header = TRUE,row.names=1)
colnames(expression_matrix)<- row.names(cellfea)

#using imputation result to cluster
HSMM_expr_matrix <- read.table('E:\\SingleCell\\imputation_0.1\\9.Chung_LTMG_0.1_0.0-0.3-0.1_recon.csv',header = FALSE,sep = ",")
colnames(HSMM_expr_matrix) <- row.names(expression_matrix)
row.names(HSMM_expr_matrix) <- colnames(expression_matrix)
exp = t(exp(as.matrix(HSMM_expr_matrix)))

#exp = as.matrix(expression_matrix)
cds <- new_cell_data_set(exp,
                         cell_metadata = cellfea,
                         gene_metadata = gene )


cds <- preprocess_cds(cds, num_dim = 50)
cds <- reduce_dimension(cds, preprocess_method = "PCA",reduction_method = c('UMAP'))
#cds <- reduce_dimension(cds, preprocess_method = "PCA")

cds = cluster_cells(cds, reduction_method = c("UMAP"),cluster_method ="louvain")

#plot_cells(cds)
#cds@clusters$UMAP$clusters
write.csv(cds@reducedDims@listData$UMAP,'E:\\SingleCell\\imputation_0.1\\ExpMonocle3Pre\\Chung_cell_pre_UMAP.csv')
write.csv(cds@clusters$UMAP$clusters,'E:\\SingleCell\\imputation_0.1\\ExpMonocle3Pre\\Chung_cell_pre_label.csv')

