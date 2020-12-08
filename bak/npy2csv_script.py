import numpy as np
import pandas as pd

def convert(method='dca'):
    t=np.load(method+'\\9.Chung_0.0_1_recon.npy')
    df = pd.DataFrame(t)
    df.to_csv(method+'_9.csv',header=None,index=False)

    t=np.load(method+'\\11.Kolodziejczyk_0.0_1_recon.npy')
    df = pd.DataFrame(t)
    df.to_csv(method+'_11.csv',header=None,index=False)

    t=np.load(method+'\\12.Klein_0.0_1_recon.npy')
    df = pd.DataFrame(t)
    df.to_csv(method+'_12.csv',header=None,index=False)

    t=np.load(method+'\\13.Zeisel_0.0_1_recon.npy')
    df = pd.DataFrame(t)
    df.to_csv(method+'_13.csv',header=None,index=False)

convert('dca')
convert('deepimpute')
convert('magic')
convert('netNMFsc')
convert('saucie')
convert('saver')
convert('scimpute')
convert('scvi')


def convertCSV(method='scIGANs'):
    df = pd.read_csv(method+'\\9.Chung_0.0_1_recon.csv.txt',sep='\s+',index_col=0)
    df = df.T
    df.to_csv(method+'_9.csv',header=None,index=False)

    df = pd.read_csv(method+'\\11.Kolodziejczyk_0.0_1_recon.csv.txt',sep='\s+',index_col=0)
    df = df.T
    df.to_csv(method+'_11.csv',header=None,index=False)

    df = pd.read_csv(method+'\\12.Klein_0.0_1_recon.csv.txt',sep='\s+',index_col=0)
    df = df.T
    df.to_csv(method+'_12.csv',header=None,index=False)

    df = pd.read_csv(method+'\\13.Zeisel_0.0_1_recon.csv.txt',sep='\s+',index_col=0)
    df = df.T
    df.to_csv(method+'_13.csv',header=None,index=False)

convertCSV('scIGANs')


