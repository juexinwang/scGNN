import argparse
parser = argparse.ArgumentParser(description='Summary on Results from Cluster output results')
parser.add_argument('--fileDir', type=str, default='', help="fileDir")
parser.add_argument('--fileName', type=str, default='results-18979328.out', help="fileName")
parser.add_argument('--outFileName', type=str, default='RC_1_0.txt', help="outFileName")
args = parser.parse_args()

fileDir = args.fileDir
fileName = args.fileName
outFileName = args.outFileName
keyDict = {'Louvain':0,'KMeans':0,'SpectralClustering':0,'AffinityPropagation':0,'AgglomerativeClustering':0,'Birch':0, 'OPTICS':0}

tabuDict =[3,4,6,7,9,10,12,13,15,16]
outLines = []
tmpstr = ''
count = 0
with open(fileDir+fileName) as f:
    lines = f.readlines()
    tag = False
    otag = False
    for line in lines:
        line = line.strip()
        if line in keyDict:            
            # if line == 'Original PCA':
            #     if keyDict[line]%18 == 0:
            #         otag = True
            #     else:
            #         otag = False
            # else:
            tag = True
            tmpstr = line+'\t'
            keyDict[line] = keyDict[line]+1
        elif tag:
            if not keyDict['Louvain'] in tabuDict:
                outLines.append(tmpstr+line+'\n')
                tag = False
                tmpstr = ''
    f.close()

with open(outFileName,'w') as fw:
    fw.writelines(outLines)
    fw.close()



