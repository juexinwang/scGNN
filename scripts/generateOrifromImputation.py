import numpy as np
import pandas as pd

i=np.load('9.Chung_LTMG_0.1_dropi.npy')
j=np.load('9.Chung_LTMG_0.1_dropj.npy')
ix=np.load('9.Chung_LTMG_0.1_dropix.npy')

ori = np.load('9.Chung_LTMG_0.1_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('9.csv',orim,delimiter=",",fmt='%10.4f')

i=np.load('11.Kolodziejczyk_LTMG_0.1_dropi.npy')
j=np.load('11.Kolodziejczyk_LTMG_0.1_dropj.npy')
ix=np.load('11.Kolodziejczyk_LTMG_0.1_dropix.npy')

ori = np.load('11.Kolodziejczyk_LTMG_0.1_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('11.csv',orim,delimiter=",",fmt='%10.4f')

i=np.load('12.Klein_LTMG_0.1_dropi.npy')
j=np.load('12.Klein_LTMG_0.1_dropj.npy')
ix=np.load('12.Klein_LTMG_0.1_dropix.npy')

ori = np.load('12.Klein_LTMG_0.1_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12.csv',orim,delimiter=",",fmt='%10.4f')

i=np.load('13.Zeisel_LTMG_0.1_dropi.npy')
j=np.load('13.Zeisel_LTMG_0.1_dropj.npy')
ix=np.load('13.Zeisel_LTMG_0.1_dropix.npy')

ori = np.load('13.Zeisel_LTMG_0.1_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('13.csv',orim,delimiter=",",fmt='%10.4f')