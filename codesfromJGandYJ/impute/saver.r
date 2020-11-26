# Usage:
# Rscript saver.r input.txt output.txt
# test if there is one argument: if not, return an error
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file)\n", call.=FALSE)
} 

library(SAVER)
inputfile = args[1]
outputfile = args[2]
raw.data <- read.csv(inputfile, header = FALSE, sep=',')
expr <- as.matrix(raw.data)
# Use 12 cores in saver
expr.saver <- saver(expr, ncores = 12, estimates.only = TRUE)
write.table(expr.saver, file=outputfile, row.names = F, col.names = F, sep = "\t")