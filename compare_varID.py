import numpy as np
import networkx as nx

edgeList = []
with open('/home/wangjue/scRNA/VarID_analysis/graph.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        words = line.split()
        edgeList.append((words[0],words[1]))
    
    f.close()

memberList = []
with open('/home/wangjue/scRNA/VarID_analysis/member.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        memberList.append(int(line)-1)    
    f.close()
    