import argparse
parser = argparse.ArgumentParser(description='Summary on Results from Cluster output results')
parser.add_argument('--fileDir', type=str, default='', help="fileDir")
parser.add_argument('--fileName', type=str, default='results-18979328.out', help="fileName")
parser.add_argument('--outFileName', type=str, default='RC_1_0.txt', help="outFileName")
args = parser.parse_args()

fileDir = args.fileDir
fileName = 'results-18979328.out'
outFileName = 'RC_1_0.txt'
keyDict = {'Louvain':None,'KMeans':None,'SpectralClustering':None,'AffinityPropagation':None,'AgglomerativeClustering':None,'Birch':None, 'OPTICS':None,'Original PCA':None, 'Proposed Method':None}

outLines = []
count = 0
with open(fileDir+fileName) as f:
    lines = f.readlines()
    tag = False
    for line in lines:
        line = line.strip()
        if line in keyDict:
            tag = True
            outLines.append(line+'\n')
        elif tag:
            outLines.append(line+'\n')
            tag = False
    f.close()

with open(outFileName,'w') as fw:
    fw.writelines(outLines)
    fw.close()



