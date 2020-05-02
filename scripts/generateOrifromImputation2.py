import numpy as np
import pandas as pd

i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_1/12.Klein_LTMG_0.3_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_1/12.Klein_LTMG_0.3_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_1/12.Klein_LTMG_0.3_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_1/12.Klein_LTMG_0.3_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.3_1.csv',orim,delimiter=",",fmt='%10.4f')


i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_2/12.Klein_LTMG_0.3_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_2/12.Klein_LTMG_0.3_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_2/12.Klein_LTMG_0.3_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_2/12.Klein_LTMG_0.3_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.3_2.csv',orim,delimiter=",",fmt='%10.4f')

i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_3/12.Klein_LTMG_0.3_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_3/12.Klein_LTMG_0.3_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_3/12.Klein_LTMG_0.3_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.3_3/12.Klein_LTMG_0.3_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.3_3.csv',orim,delimiter=",",fmt='%10.4f')


i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_1/12.Klein_LTMG_0.6_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_1/12.Klein_LTMG_0.6_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_1/12.Klein_LTMG_0.6_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_1/12.Klein_LTMG_0.6_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.6_1.csv',orim,delimiter=",",fmt='%10.4f')


i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_2/12.Klein_LTMG_0.6_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_2/12.Klein_LTMG_0.6_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_2/12.Klein_LTMG_0.6_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_2/12.Klein_LTMG_0.6_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.6_2.csv',orim,delimiter=",",fmt='%10.4f')

i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_3/12.Klein_LTMG_0.6_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_3/12.Klein_LTMG_0.6_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_3/12.Klein_LTMG_0.6_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.6_3/12.Klein_LTMG_0.6_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.6_3.csv',orim,delimiter=",",fmt='%10.4f')




i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_1/12.Klein_LTMG_0.9_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_1/12.Klein_LTMG_0.9_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_1/12.Klein_LTMG_0.9_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_1/12.Klein_LTMG_0.9_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.9_1.csv',orim,delimiter=",",fmt='%10.4f')


i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_2/12.Klein_LTMG_0.9_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_2/12.Klein_LTMG_0.9_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_2/12.Klein_LTMG_0.9_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_2/12.Klein_LTMG_0.9_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.9_2.csv',orim,delimiter=",",fmt='%10.4f')

i=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_3/12.Klein_LTMG_0.9_dropi.npy')
j=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_3/12.Klein_LTMG_0.9_dropj.npy')
ix=np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_3/12.Klein_LTMG_0.9_dropix.npy')

ori = np.load('/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_LK_0.9_3/12.Klein_LTMG_0.9_features.npy',allow_pickle=True)
ori1 = ori.tolist()
orim = ori1.todense()

for item in ix:
	orim[i[item],j[item]]=0.0

np.savetxt('12_ori_0.9_3.csv',orim,delimiter=",",fmt='%10.4f')