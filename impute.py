import numpy as np
from benchmark_util import * 

#Used to postprocess results of imputation
featuresOriginal = np.load('/home/wangjue/myprojects/scGNN/npyImpute')
features         = np.load('/home/wangjue/myprojects/scGNN/npyImpute')
dropi            = np.load('/home/wangjue/myprojects/scGNN/npyImpute')
dropj            = np.load('/home/wangjue/myprojects/scGNN/npyImpute')
dropix           = np.load('/home/wangjue/myprojects/scGNN/npyImpute')

featuresImpute   = np.load('/home/wangjue/myprojects/scGNN/npyImpute')

l1Error = imputation_error(featuresImpute, featuresOriginal, features, dropi, dropj, dropix)

print(l1Error)

