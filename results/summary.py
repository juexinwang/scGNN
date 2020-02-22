import argparse
parser = argparse.ArgumentParser(description='Summary on Results from Cluster output results')
parser.add_argument('--fileDir', type=str, default='', help="fileDir")
parser.add_argument('--fileName', type=str, default='results-19687315.out', help="fileName")
parser.add_argument('--outFileName', type=str, default='RC_0_2.txt', help="outFileName")
args = parser.parse_args()

#TODO implement late
fileDir = args.fileDir
fileName = args.fileName
outFileName = args.outFileName
keyDict = {'Louvain':0,'KMeans':0,'SpectralClustering':0,'AffinityPropagation':0,'AgglomerativeClustering':0,'Birch':0, 'OPTICS':0}

# tabuDict =[1,2,3,6,9,12,15,18]
tabuDict =[0,1,2,5,8,11,14,17,20,23,26,29,32]
outLines = []
tmpstr = ''
count = 0
with open(fileDir+fileName) as f:
    lines = f.readlines()
    tag = False
    otag = False
    convergetag = False
    for line in lines:
        line = line.strip()
        if line in keyDict:            
            tag = True
            tmpstr = '\t'+line+'\t'
            if line == 'Louvain':
                val = keyDict['Louvain']%33
                if val == 0:
                    priorstr = 'Graph'
                elif val == 1:
                    priorstr = 'NoGraph'
                elif val == 2:
                    priorstr = 'Z'
                elif val == 5:
                    priorstr = 'Z0'
                elif val == 8:
                    priorstr = 'Z1'
                elif val == 11:
                    priorstr = 'Z2'
                elif val == 14:
                    priorstr = 'Z3'
                elif val == 17:
                    priorstr = 'Z4'
                elif val == 20:
                    priorstr = 'Z5'
                elif val == 23:
                    priorstr = 'Z6'
                elif val == 26:
                    priorstr = 'Z7'
                elif val == 29:
                    priorstr = 'Z8'
                elif val == 32:
                    priorstr = 'Z9'
                tmpstr = priorstr + tmpstr 
            keyDict[line] = keyDict[line]+1        
        elif line.startswith('FileNotFoundError: [Errno 2] No such file or directory:'):
            print(line)
            break
        elif line.endswith('?it/s]'):
            tag = False
            tmpstr = ''
        elif tag:
            val = (keyDict['Louvain']-1)%33
            if val in tabuDict:
                outLines.append(tmpstr+line+'\n')
                tag = False
                tmpstr = ''
    f.close()

with open(outFileName,'w') as fw:
    fw.writelines(outLines)
    fw.close()



