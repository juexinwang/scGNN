from multiprocessing import Pool
import numpy as np
import time
from scipy.spatial import distance_matrix, minkowski_distance, distance
import resource

class CelltypeAEParallel():
    '''
    Celltype AutoEncoder training in parallel
    '''
    # def __init__(self,recon,clusterIndexList,args):
    #     self.recon = recon
    #     self.clusterIndexList = clusterIndexList   
    #     self.batch_size = args.batch_size
    #     self.celltype_epochs = args.celltype_epochs

    def __init__(self,recon,clusterIndexList):
        self.recon = recon
        self.clusterIndexList = clusterIndexList  

    def trainParallel(self,i):
        '''
        Train each autoencoder in paral
        '''
        clusterIndex = self.clusterIndexList[i]

        reconUsage = self.recon[clusterIndex]
        reconCluster = reconUsage + 1.0
        # scDataInter = scDatasetInter(reconUsage)
        # train_loader = DataLoader(scDataInter, batch_size=self.batch_size, shuffle=False, **kwargs)
        # for epoch in range(1, self.celltype_epochs + 1):
        #     reconCluster, originalCluster, zCluster = train(epoch, EMFlag=True)                
        
        return reconCluster
    
    def work(self):
        return Pool().map(self.trainParallel, range(len(self.clusterIndexList)))

if __name__ == "__main__":

    recon = np.random.rand(10,5)
    reconNew = np.copy(recon)
    clusterIndexList = []

    for i in range(3):
        clusterIndexList.append([])

    clusterIndexList[0].append(0)
    clusterIndexList[0].append(5)
    clusterIndexList[0].append(7)

    clusterIndexList[1].append(1)
    clusterIndexList[1].append(2)
    clusterIndexList[1].append(6)

    clusterIndexList[2].append(3)
    clusterIndexList[2].append(4)
    clusterIndexList[2].append(8)
    clusterIndexList[2].append(9)
    # tmp={}
    # tmp[0]=0
    # tmp[1]=5
    # tmp[2]=7
    # clusterIndexList.append(tmp)
    # tmp={}
    # tmp[0]=1
    # tmp[1]=2
    # tmp[2]=6
    # clusterIndexList.append(tmp)
    # tmp[0]=3
    # tmp[1]=4
    # tmp[2]=8
    # tmp[3]=9
    # clusterIndexList.append(tmp)

    print(recon)

    with Pool() as p:
        # reconp = CelltypeAEParallel(recon,clusterIndexList,args).work()
        reconp = CelltypeAEParallel(recon,clusterIndexList).work()

        for index in range(len(clusterIndexList)):
            count = 0
            clist = clusterIndexList[index]
            for i in clist:
                reconNew[i] = reconp[index][count,:]
                count +=1
    
    print(reconNew)
    print(3)
