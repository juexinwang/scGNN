# This code has not cleaned yet
# run netNMF-sc from command line and save outputs to specified directory
from __future__ import print_function
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
import copy,argparse,os,math,random,time
from scipy import sparse, io,linalg
from scipy.sparse import csr_matrix
import warnings,os
from netNMFsc import plot
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

def main(args):
    if args.method == 'GD':
        from netNMFsc import netNMFGD
        operator = netNMFGD(d=args.dimensions, alpha=args.alpha, n_inits=1, tol=args.tol, max_iter=args.max_iters, n_jobs=4)
    elif args.method == 'MU':
        from netNMFsc import netNMFMU
        operator = netNMFMU(d=args.dimensions, alpha=args.alpha, n_inits=1, tol=args.tol, max_iter=args.max_iters, n_jobs=4)

    filename = '/storage/hpc/group/joshilab/scGNNdata/{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(
        args.Randomdata, args.datasetName,args.dropratio)
    x = np.load(filename, allow_pickle=True)
    x = x.tolist()
    x = x.todense()
    x = np.asarray(x)
    if args.process == 'log':
        x = np.log(x + 1)

    # transpose and add names for rows and cols
    features = np.transpose(x)

    chung = pd.read_csv(args.filename, header=0,
                        index_col=0, sep=',')
    X = features
    genes = []
    for gen in chung.index.values:
        if '.' in gen:
            genes.append(gen.upper().split('.')[0])
        else:
            genes.append(gen.upper())
    #print(genes)
    operator.genes = np.asarray(genes)
    operator.X = X
    #operator.load_10X(direc=args.tenXdir,genome='mm10')
    operator.load_network(net=args.network,genenames=args.netgenes,sparsity=args.sparsity)
    dictW = operator.fit_transform()
    W, H = dictW['W'], dictW['H']
    # k,clusters = plot.select_clusters(H,max_clusters=20)
    # plot.tSNE(H,clusters,fname=args.direc+ '/netNMFsc_tsne_imputation_' +args.process +'_'+args.Randomdata)
    # os.system('mkdir -p %s'%(args.direc))
    np.save(os.path.join(args.direc,args.Randomdata+'_'+args.process+'_imputation.npy'),np.dot(W,H))
    #np.save(os.path.join(args.direc,'H.npy'),H)
    #np.save(os.path.join(args.direc, 'cluster.npy'), H)
    return
#/storage/htc/joshilab/jghhd/singlecellTest/netNMFsc/netNMF-sc/netNMFsc/refdata/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--method",help="either 'GD for gradient descent or MU for multiplicative update",type=str,default='GD')
    parser.add_argument("-f","--filename", help="path to data file (.npy or .mtx)",type=str,default='matrix.mtx')
    parser.add_argument("-g","--gene_names", help="path to file containing gene names (.npy or .tsv)",type=str,default='gene_names.tsv')
    parser.add_argument("-net","--network", help="path to network file (.npy or .mtx)",type=str,default='')
    parser.add_argument("-netgenes","--netgenes", help="path to file containing gene names for network (.npy or .tsv)",type=str,default='')
    parser.add_argument("-org","--organism", help="mouse or human",type=str,default='human')
    parser.add_argument("-id","--idtype", help="ensemble, symbol, or entrez",type=str,default='ensemble')
    parser.add_argument("-netid","--netidtype", help="ensemble, symbol, or entrez",type=str,default='entrez')
    parser.add_argument("-n","--normalize", help="normalize data? 1 = yes, 0 = no",type=int,default=0)
    parser.add_argument("-sparse","--sparsity", help="sparsity for network",type=float,default=0.99)
    parser.add_argument("-mi","--max_iters", help="max iters for netNMF-sc",type=int,default=1500)
    parser.add_argument("-t","--tol", help="tolerence for netNMF-sc",type=float,default=1e-2)
    parser.add_argument("-d","--direc", help="directory to save files",default='')
    parser.add_argument("-D","--dimensions", help="number of dimensions to apply shift",type=int,default = 10)
    parser.add_argument("-a","--alpha", help="lambda param for netNMF-sc",type=float,default = 1.0)
    parser.add_argument("-x","--tenXdir", help="data is from 10X. Only required to provide directory containing matrix.mtx, genes.tsv, barcodes.tsv files",type=str,default = '')
    parser.add_argument('--Randomdata', type=str, default='npyImputeG2E_1', help='npyImputeG2E_1,2,3')
    parser.add_argument('--datasetName', type=str, default='12.Klein', help='12.Klein,13.Zeisel')
    parser.add_argument('--process', type=str, default='null', help='log/null to process data')
    parser.add_argument("-Hasdot","--Hasdot",type = bool, help="data gene names has dot",default = True)
    parser.add_argument('--dropratio', type=str, default='0.1', help='0.1，0.3，0.6，0.8')
    args = parser.parse_args()
    main(args)


#'/storage/htc/joshilab/jghhd/singlecellTest/Data/11.Kolodziejczyk/Use_expression.csv'
