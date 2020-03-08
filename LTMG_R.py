import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('scGNNLTMG')

def runLTMG(dir,expressionfile,ltmgfile):
    # robjects.r('''
    #         setwd("/users/PAS1475/qiren081/GCNN/data/sc/ex")
    #         test.data <- read.csv("Biase_expression.csv",header = T,row.names = 1,check.names = F)
    #         object <- scGNN.LTMG::CreateLTMGObject(as.matrix(test.data))
    #         object <- scGNN.LTMG::RunLTMG(object,Gene_use = "all",seed =123,k=5)
    #         my.matrix <- cbind(ID = rownames(object@OrdinalMatrix),object@OrdinalMatrix)
    #         write.table(my.matrix, file = "LTMG_discretization_Bia.txt",row.names = F, quote = F,sep = "\t")
    # ''')
    robjects.r('''
            setwd("%s")
            test.data <- read.csv("%s",header = T,row.names = 1,check.names = F)
            object <- scGNN.LTMG::CreateLTMGObject(as.matrix(test.data))
            object <- scGNN.LTMG::RunLTMG(object,Gene_use = "all",seed =123,k=5)
            my.matrix <- cbind(ID = rownames(object@OrdinalMatrix),object@OrdinalMatrix)
            write.table(my.matrix, file = "%s",row.names = F, quote = F,sep = "\t")
    ''')%(dir,expressionfile,ltmgfile)