import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

igraph = importr('igraph')
base   = importr('base')
fromV  = ro.FloatVector([0,0,0,3])
toV    = ro.FloatVector([1,2,3,4])
# weightV= ro.FloatVector([0.1,1.0,1.0,0.1])
weightV= ro.FloatVector([1.0,1.0,1.0,1.0])
links  = ro.DataFrame({'from':fromV,'to':toV,'weight':weightV})
g  = igraph.graph_from_data_frame(links,directed = False)
cl = igraph.cluster_louvain(g)

def as_dict(vector):
    """Convert an RPy2 ListVector to a Python dict"""
    result = {}
    for i, name in enumerate(vector.names):
        if isinstance(vector[i], ro.ListVector):
            result[name] = as_dict(vector[i])
        elif len(vector[i]) == 1:
            result[name] = vector[i][0]
        else:
            result[name] = vector[i]
    return result

cl_dict = as_dict(cl)
cl_dict['membership']


# R code:
# library(igraph)

# fromV  <- as.vector(c(0,0,0,3))
# toV    <- as.vector(c(1,2,3,4))
# weightV<- as.vector(c(0.1,1.0,1.0,0.1))
# links  <- data.frame(from=fromV,to=toV,weight=weightV)
# g  <-  graph_from_data_frame(links,directed = FALSE)
# cl <- cluster_louvain(g)