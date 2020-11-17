# R
# Running after plot_distribution.py

# http://www.gamlss.com/wp-content/uploads/2013/01/book-2010-Athens1.pdf
# https://arxiv.org/pdf/1810.02618.pdf
# https://rdrr.io/cran/gamlss.dist/man/ZANBI.html

#install in conda:
# https://anaconda.org/conda-forge/r-fitdistrplus
# https://anaconda.org/conda-forge/r-gamlss
# install.packages("fitdistrplus")
# install.packages("gamlss")
library(fitdistrplus)
library(gamlss)

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least four argument must be supplied (input file).n", call.=FALSE)
}

datasetName=args[1]
para=args[2]
indir=args[3]
outdir=args[4]

features = read.table(paste(indir,"/",datasetName,"_",para,"_features.txt",sep=''), header = FALSE, sep = " ")
features = data.matrix(features)
features = as.vector(features)
features = as.numeric(features)

print(paste(indir,"/",datasetName,"_",para,"_features.txt",sep=''))
mu_ = mean(features)
sigma_ = (sd(features)-mean(features))/mean(features)**2
# http://www.gamlss.com/wp-content/uploads/2013/01/book-2010-Athens1.pdf Page 219
fit_nbi = fitdist(features, 'NBI',   start = list(mu = mu_, sigma = sigma_ ))
gofstat(fit_nbi)
tiff(file= paste(outdir,"/",datasetName,"_",para,"_NBI.tiff",sep=''))
plot(fit_nbi)
dev.off()

# http://www.gamlss.com/wp-content/uploads/2013/01/book-2010-Athens1.pdf Page 221
nu_ = 1-length(which(features!=0))/(length(features))
fit_zinb= fitdist(features, 'ZINBI', start = list(mu = mu_, sigma = sigma_, nu = nu_))
gofstat(fit_zinb)
tiff(file=paste(outdir,"/",datasetName,"_",para,"_ZINBI.tiff",sep=''))
plot(fit_zinb)
dev.off()

fit_zinb_= fitdist(features, 'ZINBI', start = list(mu = mu_, sigma = sigma_))
gofstat(fit_zinb_)
tiff(file=paste(outdir,"/",datasetName,"_",para,"_ZINBI.tiff_",sep=''))
plot(fit_zinb_)
dev.off()


# NBI:
# Goodness-of-fit statistics
#                                 1-mle-NBI
# Kolmogorov-Smirnov statistic 3.671374e-01
# Cramer-von Mises statistic   1.016737e+05
# Anderson-Darling statistic            Inf

# Goodness-of-fit criteria
#                                1-mle-NBI
# Akaike's Information Criterion  25429885
# Bayesian Information Criterion  25429912


# ZINB
# Goodness-of-fit statistics
#                               1-mle-ZINBI
# Kolmogorov-Smirnov statistic 4.532250e-01
# Cramer-von Mises statistic   1.873046e+05
# Anderson-Darling statistic            Inf

# Goodness-of-fit criteria
#                                1-mle-ZINBI
# Akaike's Information Criterion    25969108
# Bayesian Information Criterion    25969135