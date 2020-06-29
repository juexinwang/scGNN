# Calculate ROGUE value for clustering results
# Ref: https://github.com/PaulingLiu/ROGUE
# Citation: Liu, Baolin, Chenwei Li, Ziyi Li, Dongfang Wang, Xianwen Ren, and Zemin Zhang. "An entropy-based metric for assessing the purity of single cell populations." Nature Communications 11, no. 1 (2020): 1-13.

#Use enviorment in HPC
#source activate env-tidyverse

suppressMessages(library(ROGUE))
suppressMessages(library(ggplot2))
suppressMessages(library(tidyverse))

# expr=read.csv("/storage/htc/joshilab/wangjue/imputed/all/12.magic.1.csv", header = FALSE)
# ent.res <- SE_fun(expr)
# CalculateRogue(ent.res, platform = "UMI")
# rogue(expr1, labels = cluster, samples = sample, platform = "UMI", span = 0.6)

# For Dataset Klein
expr=read.csv("12.data.csv", header = FALSE)

sample =rep('p',dim(expr)[1])
expr=t(expr)

# Whether use original expression data
# expr=exp(expr)

# CIDR
clusters=read.csv("CIDR/12.Klein_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# monocle3
clusters=read.csv("monocle3/Klein_cell_pre_label.csv")
clusters=unlist(clusters[2], use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# phenoGraph
clusters=read.csv("phenoGraph/Klein_cell_pre_label.csv")
clusters=unlist(clusters[2], use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# raceID
clusters=read.csv("raceID/12.Klein_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# seurat
clusters=read.csv("seurat/12.Klein_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# toomanycells
clusters=read.csv("toomanycells/12.Klein_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# scGNN
clusters=read.csv("/storage/htc/joshilab/wangjue/scGNN/npyG2E_LK_1/12.Klein_LTMG_0.0-0.3-0.1_0.0_0.0_results.txt", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)


# For dataset Zesel
expr=read.csv("13.data.csv", header = FALSE)
sample =rep('p',dim(expr)[1])
expr=t(expr)

# Whether use original expression data
# expr=exp(expr)

# CIDR
clusters=read.csv("CIDR/13.Zeisel_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# monocle3
clusters=read.csv("monocle3/Zeisel_cell_pre_label.csv")
clusters=unlist(clusters[2], use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# phenoGraph
clusters=read.csv("phenoGraph/Zeisel_cell_pre_label.csv")
clusters=unlist(clusters[2], use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# raceID
clusters=read.csv("raceID/13.Zeisel_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# seurat
clusters=read.csv("seurat/13.Zeisel_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# toomanycells
clusters=read.csv("toomanycells/13.Zeisel_clusters.csv", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
# scGNN
clusters=read.csv("/storage/htc/joshilab/wangjue/scGNN/npyG2E_LK_1/13.Zeisel_LTMG_0.0-0.3-0.1_0.0_0.0_results.txt", header = FALSE)
clusters=unlist(clusters, use.names=FALSE)
rogue(expr, labels = clusters, samples = sample, platform = "UMI", span = 0.6)
